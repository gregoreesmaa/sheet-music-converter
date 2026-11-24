"""MuseScore (.mscx/.mscz) & MusicXML parser + page layout + MVP handwritten renderer.

Integrates `styles.py`, `layout_quirks.py`, `stroke_engine.py`, and `glyph_variants.py`
to render real scores into any of the 20 handwritten scribe styles with exact ground-truth
Bezier staff metadata JSON.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import random
import sys
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

# Allow running both as `python -m hw_engine.render_mvp` and directly as a script
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# ---------------------------------------------------------------------------
# Score Data Model
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
    "G-2": 30,  # E4 on line 1 (G4 on line 2)
    "F-4": 18,  # G2 on line 1 (F3 on line 4)
    "C-3": 24,  # F3 on line 1 (C4 on line 3)
    "C-4": 22,  # D3 on line 1 (C4 on line 4)
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

    @property
    def beats(self) -> float:
        base = DURATION_BEATS.get(self.duration_kind, 1.0)
        mult = 1.0
        add = 0.5
        for _ in range(self.dots):
            mult += add
            add *= 0.5
        return base * mult


@dataclass
class RestEvent:
    """Rest at a rhythmic position."""

    duration_kind: str  # 'whole', 'half', 'quarter', 'eighth', '16th', '32nd'
    dots: int = 0
    beat_offset: float = 0.0
    is_measure_rest: bool = False

    @property
    def beats(self) -> float:
        base = DURATION_BEATS.get(self.duration_kind, 1.0)
        mult = 1.0
        add = 0.5
        for _ in range(self.dots):
            mult += add
            add *= 0.5
        return base * mult


ScoreEvent = Union[NoteEvent, RestEvent]


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


@dataclass
class PartData:
    """One notated staff (e.g. Soprano, Piano RH, Piano LH)."""

    name: str
    staff_index: int
    measures: List[MeasureData] = field(default_factory=list)
    brace_group: Optional[int] = None  # Staves sharing a non-None brace_group get a brace


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
# Helpers for Pitch, Duration, and Beaming
# ---------------------------------------------------------------------------


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
        "sharp2": "double_sharp",
        "flat2": "double_flat",
        "accidentalsharp": "sharp",
        "accidentalflat": "flat",
        "accidentalnatural": "natural",
        "accidentaldoublesharp": "double_sharp",
        "accidentaldoubleflat": "double_flat",
    }
    return mapping.get(k)


def _diatonic_to_staff_step(letter: str, octave: int, clef: str) -> int:
    diatonic = octave * 7 + LETTER_TO_DIATONIC.get(letter.upper(), 0)
    bottom = CLEF_BOTTOM_DIATONIC.get(clef, 30)
    return diatonic - bottom


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
    # Fallback without TPC
    pc_map = [
        ("C", 0), ("C", 1), ("D", 0), ("E", -1), ("E", 0), ("F", 0),
        ("F", 1), ("G", 0), ("A", -1), ("A", 0), ("B", -1), ("B", 0),
    ]
    letter, alter = pc_map[pitch % 12]
    octave = (pitch // 12) - 1
    return letter, octave, alter


def _assign_auto_beams(events: List[ScoreEvent], time_sig: Tuple[int, int]) -> None:
    """Assign beat-grouped beam_group IDs to beamable NoteEvents lacking explicit beams."""
    if any(isinstance(e, NoteEvent) and e.beam_group is not None for e in events):
        return
    num, den = time_sig
    # Beat size in quarter notes
    if den == 8 and num in (6, 9, 12):
        group_beats = 1.5
    elif den == 2:
        group_beats = 2.0
    else:
        group_beats = 1.0

    beamable = {"eighth", "16th", "32nd"}
    group_id = 0
    run: List[NoteEvent] = []
    run_bucket: Optional[int] = None

    def flush() -> None:
        nonlocal group_id, run
        if len(run) >= 2:
            for ev in run:
                ev.beam_group = group_id
            group_id += 1
        run = []

    for ev in events:
        if isinstance(ev, NoteEvent) and ev.duration_kind in beamable:
            bucket = int(math.floor((ev.beat_offset + 1e-4) / group_beats))
            if run_bucket is not None and bucket != run_bucket:
                flush()
            run_bucket = bucket
            run.append(ev)
        else:
            flush()
            run_bucket = None
    flush()


# ---------------------------------------------------------------------------
# MusicXML Parser (via music21)
# ---------------------------------------------------------------------------


def _m21_clef_code(clef_obj: Any) -> str:
    if clef_obj is None:
        return "G-2"
    sign = getattr(clef_obj, "sign", "G") or "G"
    line = getattr(clef_obj, "line", 2) or 2
    code = f"{sign}-{line}"
    if code in CLEF_BOTTOM_DIATONIC:
        return code
    if sign == "F":
        return "F-4"
    if sign == "C":
        return "C-3" if line == 3 else "C-4"
    return "G-2"


def _m21_barline_type(measure_obj: Any) -> str:
    rb = getattr(measure_obj, "rightBarline", None)
    if rb is None:
        return "single"
    cls_name = type(rb).__name__.lower()
    if "repeat" in cls_name:
        return "repeat_right"
    btype = (getattr(rb, "type", "") or "").lower()
    if btype in ("final", "light-heavy"):
        return "final"
    if btype in ("double", "light-light"):
        return "double"
    return "single"


def _parse_musicxml(path: Path) -> ParsedScore:
    import music21

    if path.suffix.lower() == ".mxl":
        xml_text: Optional[str] = None
        with zipfile.ZipFile(path, "r") as zf:
            for name in zf.namelist():
                if name.lower().endswith((".musicxml", ".xml")) and not name.startswith("META-INF"):
                    xml_text = zf.read(name).decode("utf-8", errors="replace")
                    break
        if xml_text is None:
            raise ValueError(f"No MusicXML document found inside {path}")
        score = music21.converter.parseData(xml_text)
    else:
        # Use parseData to avoid music21 temp-cache permission issues on Windows
        xml_text = path.read_text(encoding="utf-8", errors="replace")
        score = music21.converter.parseData(xml_text)

    title = path.stem
    if score.metadata and getattr(score.metadata, "title", None):
        title = str(score.metadata.title)

    # Identify brace groups from StaffGroup layout objects
    part_to_brace: Dict[int, int] = {}
    parts_list = list(score.parts)
    for g_idx, sg in enumerate(score.recurse().getElementsByClass(music21.layout.StaffGroup)):
        if getattr(sg, "symbol", None) == "brace":
            for sp in sg.getSpannedElements():
                for p_idx, p in enumerate(parts_list):
                    if p is sp:
                        part_to_brace[p_idx] = g_idx

    # Heuristic: if 2 parts share "Piano" in name and no brace group was found, brace them
    if not part_to_brace and len(parts_list) == 2:
        names = [(getattr(p, "partName", "") or "").lower() for p in parts_list]
        if any("piano" in n for n in names):
            part_to_brace[0] = 0
            part_to_brace[1] = 0

    parsed_parts: List[PartData] = []
    for p_idx, part in enumerate(parts_list):
        part_name = getattr(part, "partName", None) or f"Part {p_idx + 1}"
        measures_out: List[MeasureData] = []
        active_clef = "G-2"
        active_key = 0
        active_ts = (4, 4)

        m_list = list(part.getElementsByClass(music21.stream.Measure))
        for m_idx, m in enumerate(m_list):
            clef_change = False
            key_change = False
            time_change = False

            clefs = list(m.getElementsByClass(music21.clef.Clef))
            if clefs:
                new_clef = _m21_clef_code(clefs[0])
                if new_clef != active_clef or m_idx == 0:
                    clef_change = m_idx > 0 and new_clef != active_clef
                    active_clef = new_clef

            keys = list(m.getElementsByClass(music21.key.KeySignature))
            if keys:
                new_key = int(getattr(keys[0], "sharps", 0))
                if new_key != active_key:
                    key_change = m_idx > 0
                    active_key = new_key

            tss = list(m.getElementsByClass(music21.meter.TimeSignature))
            if tss:
                new_ts = (int(tss[0].numerator), int(tss[0].denominator))
                if new_ts != active_ts:
                    time_change = m_idx > 0
                    active_ts = new_ts

            # Pick first voice if polyphonic voices exist, else direct measure elements
            voices = list(m.getElementsByClass(music21.stream.Voice))
            source_stream = voices[0] if voices else m

            events: List[ScoreEvent] = []
            beam_counter = 0
            in_beam = False

            for el in source_stream.notesAndRests:
                beat_off = float(getattr(el, "offset", 0.0))
                dur_obj = el.duration
                dur_kind = _normalize_duration_kind(getattr(dur_obj, "type", "quarter"))
                dots = int(getattr(dur_obj, "dots", 0))

                if isinstance(el, music21.note.Rest):
                    is_mr = bool(getattr(el, "fullMeasure", False)) or (
                        dur_kind == "whole" and len(list(source_stream.notesAndRests)) == 1
                    )
                    events.append(
                        RestEvent(
                            duration_kind="whole" if is_mr else dur_kind,
                            dots=dots,
                            beat_offset=beat_off,
                            is_measure_rest=is_mr,
                        )
                    )
                    in_beam = False
                elif isinstance(el, (music21.note.Note, music21.chord.Chord)):
                    pitches = [el.pitch] if isinstance(el, music21.note.Note) else list(el.pitches)
                    step_acc: List[Tuple[int, Optional[str]]] = []
                    for pit in pitches:
                        st = _diatonic_to_staff_step(pit.step, pit.octave, active_clef)
                        acc_str: Optional[str] = None
                        if pit.accidental is not None:
                            disp = pit.accidental.displayStatus
                            if disp is True or disp is None:
                                acc_str = _normalize_accidental(pit.accidental.name)
                        step_acc.append((st, acc_str))
                    step_acc.sort(key=lambda t: t[0])

                    beam_grp: Optional[int] = None
                    beams = getattr(el, "beams", None)
                    if beams and beams.beamsList:
                        btype = beams.beamsList[0].type
                        if btype == "start":
                            in_beam = True
                            beam_grp = beam_counter
                        elif btype == "continue" and in_beam:
                            beam_grp = beam_counter
                        elif btype == "stop" and in_beam:
                            beam_grp = beam_counter
                            beam_counter += 1
                            in_beam = False
                    else:
                        in_beam = False

                    events.append(
                        NoteEvent(
                            staff_steps=[s for s, _ in step_acc],
                            accidentals=[a for _, a in step_acc],
                            duration_kind=dur_kind,
                            dots=dots,
                            beat_offset=beat_off,
                            beam_group=beam_grp,
                        )
                    )

            _assign_auto_beams(events, active_ts)
            barline = _m21_barline_type(m)
            m_num = int(getattr(m, "number", m_idx + 1) or (m_idx + 1))
            measures_out.append(
                MeasureData(
                    number=m_num,
                    clef=active_clef,
                    key_fifths=active_key,
                    time_sig=active_ts,
                    clef_change=clef_change,
                    key_change=key_change,
                    time_change=time_change,
                    barline=barline,
                    events=events,
                )
            )

        parsed_parts.append(
            PartData(
                name=part_name,
                staff_index=p_idx,
                measures=measures_out,
                brace_group=part_to_brace.get(p_idx),
            )
        )

    return ParsedScore(title=title, source_path=str(path), parts=parsed_parts)


# ---------------------------------------------------------------------------
# MuseScore (.mscx / .mscz) XML Parser (Zero External Binary)
# ---------------------------------------------------------------------------


def _mscx_clef_code(raw: Optional[str]) -> str:
    if not raw:
        return "G-2"
    r = raw.strip().upper()
    if r.startswith("F"):
        return "F-4"
    if r.startswith("C4"):
        return "C-4"
    if r.startswith("C"):
        return "C-3"
    return "G-2"


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

        for m_idx, m_el in enumerate(staff_el.findall("Measure")):
            clef_change = False
            key_change = False
            time_change = False
            barline = "single"
            events: List[ScoreEvent] = []

            # MuseScore 3/4 puts items inside <voice>, MuseScore 2 puts them directly in <Measure>
            voices = m_el.findall("voice")
            container = voices[0] if voices else m_el

            beat_cursor = 0.0
            for child in container:
                tag = child.tag
                if tag == "Clef":
                    ctype = (
                        child.findtext("concertClefType")
                        or child.findtext("transposingClefType")
                        or child.findtext("subtype")
                    )
                    new_clef = _mscx_clef_code(ctype)
                    if new_clef != active_clef:
                        clef_change = m_idx > 0
                        active_clef = new_clef
                elif tag == "KeySig":
                    acc_txt = child.findtext("accidental") or child.findtext("concertKey")
                    if acc_txt is not None:
                        try:
                            new_key = max(-7, min(7, int(acc_txt)))
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
                    elif "repeat" in sub:
                        barline = "repeat_right"
                elif tag == "Rest":
                    raw_dur = child.findtext("durationType") or "quarter"
                    dots = int(child.findtext("dots") or "0")
                    is_mr = raw_dur.strip().lower() == "measure"
                    dur_kind = "whole" if is_mr else _normalize_duration_kind(raw_dur)
                    ev = RestEvent(
                        duration_kind=dur_kind,
                        dots=dots,
                        beat_offset=beat_cursor,
                        is_measure_rest=is_mr,
                    )
                    events.append(ev)
                    beat_cursor += (
                        active_ts[0] * (4.0 / active_ts[1]) if is_mr else ev.beats
                    )
                elif tag == "Chord":
                    raw_dur = child.findtext("durationType") or "quarter"
                    dots = int(child.findtext("dots") or "0")
                    dur_kind = _normalize_duration_kind(raw_dur)
                    step_acc: List[Tuple[int, Optional[str]]] = []
                    for note_el in child.findall("Note"):
                        p_txt = note_el.findtext("pitch")
                        t_txt = note_el.findtext("tpc")
                        if p_txt is None:
                            continue
                        midi_p = int(p_txt)
                        tpc = int(t_txt) if t_txt is not None else None
                        letter, octave, _alter = _mscx_pitch_tpc_to_letter_octave_alter(
                            midi_p, tpc
                        )
                        st = _diatonic_to_staff_step(letter, octave, active_clef)
                        acc_el = note_el.find("Accidental")
                        acc_str: Optional[str] = None
                        if acc_el is not None:
                            acc_str = _normalize_accidental(acc_el.findtext("subtype"))
                        step_acc.append((st, acc_str))
                    if step_acc:
                        step_acc.sort(key=lambda t: t[0])
                        ev = NoteEvent(
                            staff_steps=[s for s, _ in step_acc],
                            accidentals=[a for _, a in step_acc],
                            duration_kind=dur_kind,
                            dots=dots,
                            beat_offset=beat_cursor,
                        )
                        events.append(ev)
                        beat_cursor += ev.beats

            _assign_auto_beams(events, active_ts)
            if m_idx == len(staff_el.findall("Measure")) - 1 and barline == "single":
                barline = "final"
            measures_out.append(
                MeasureData(
                    number=m_idx + 1,
                    clef=active_clef,
                    key_fifths=active_key,
                    time_sig=active_ts,
                    clef_change=clef_change,
                    key_change=key_change,
                    time_change=time_change,
                    barline=barline,
                    events=events,
                )
            )

        parsed_parts.append(
            PartData(
                name=part_name,
                staff_index=s_idx,
                measures=measures_out,
                brace_group=brace_group,
            )
        )

    return ParsedScore(title=title, source_path=source_path, parts=parsed_parts)


def parse_score(path: Union[str, Path]) -> ParsedScore:
    """Parse a `.musicxml`, `.mxl`, `.mscx`, or `.mscz` score into a `ParsedScore`."""
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
# Page & System Layout
# ---------------------------------------------------------------------------


@dataclass
class StaffLayout:
    """Geometry for a single 5-line staff within a system."""

    part_index: int
    part_name: str
    brace_group: Optional[int]
    x_left: float
    x_right: float
    y_center: float
    sp: float
    ctrl_pts: List[List[float]] = field(default_factory=list)


@dataclass
class SystemLayout:
    """One system (horizontal row of 1..N staves) covering measures [m_start, m_end)."""

    system_index: int
    measure_start: int
    measure_end: int
    x_left: float
    x_right: float
    bar_x_bounds: List[Tuple[float, float]]  # (x0, x1) for each measure in system
    staves: List[StaffLayout] = field(default_factory=list)
    is_blank_prerule: bool = False


def layout_score_pages(
    score: ParsedScore,
    page_w: int = 1650,
    page_h: int = 2200,
    style: Optional[Dict[str, Any]] = None,
    rng: Optional[random.Random] = None,
    max_parts: int = 4,
) -> List[SystemLayout]:
    """Wrap score measures into systems that fit on page 1 (MVP single-page layout)."""
    style = style or {}
    rng = rng or random.Random(42)

    parts = score.parts[:max_parts]
    n_staves = max(1, len(parts))
    total_measures = score.num_measures

    # Staff spacing (sp) adapts slightly to part count & style
    sp_base = float(style.get("sp", style.get("staff_space", 16.0)))
    if n_staves >= 4:
        sp = min(sp_base, 12.0)
    elif n_staves == 3:
        sp = min(sp_base, 13.5)
    elif n_staves == 2:
        sp = min(sp_base, 15.0)
    else:
        sp = sp_base

    margin_x = float(style.get("margin_x", page_w * 0.065))
    margin_top = float(style.get("margin_top", page_h * 0.075))
    margin_bottom = float(style.get("margin_bottom", page_h * 0.065))
    x_left = margin_x
    x_right = page_w - margin_x

    staff_h = 4.0 * sp
    intra_gap = 6.5 * sp if n_staves > 1 else 0.0
    inter_system_gap = 7.5 * sp if n_staves == 1 else 6.0 * sp
    system_h = n_staves * staff_h + max(0, n_staves - 1) * intra_gap

    # Measures per system (4-5 typically)
    bars_per_sys_default = 4 if n_staves >= 2 else 5
    bar_width_var = float(style.get("bar_width_var", 0.12))
    prerule_prob = float(style.get("blank_prerule_prob", 0.0))

    systems: List[SystemLayout] = []
    y_cursor = margin_top + staff_h * 0.5
    m_cursor = 0
    sys_idx = 0

    while y_cursor + system_h - staff_h * 0.5 <= page_h - margin_bottom:
        if m_cursor < total_measures:
            bars_this_sys = min(
                total_measures - m_cursor,
                bars_per_sys_default + (rng.choice([-1, 0, 0, 1]) if total_measures - m_cursor > 5 else 0),
            )
            bars_this_sys = max(1, bars_this_sys)
            m_start = m_cursor
            m_end = m_cursor + bars_this_sys
            m_cursor = m_end

            # Compute per-bar widths with slight handwritten variance + note density weight
            weights: List[float] = []
            for m_i in range(m_start, m_end):
                max_ev = 4
                for p in parts:
                    if m_i < len(p.measures):
                        max_ev = max(max_ev, len(p.measures[m_i].events))
                density_w = 0.65 + 0.12 * min(max_ev, 16)
                jitter = 1.0 + rng.uniform(-bar_width_var, bar_width_var)
                # First measure of system has clef/key/time prefix
                prefix_bonus = 0.35 if m_i == m_start else 0.0
                weights.append(max(0.4, (density_w + prefix_bonus) * jitter))

            w_sum = sum(weights)
            bar_bounds: List[Tuple[float, float]] = []
            bx = x_left
            avail_w = x_right - x_left
            for w in weights:
                bw = avail_w * (w / w_sum)
                bar_bounds.append((bx, bx + bw))
                bx += bw

            staves: List[StaffLayout] = []
            for s_i, p in enumerate(parts):
                yc = y_cursor + s_i * (staff_h + intra_gap)
                staves.append(
                    StaffLayout(
                        part_index=s_i,
                        part_name=p.name,
                        brace_group=p.brace_group,
                        x_left=x_left,
                        x_right=x_right,
                        y_center=yc,
                        sp=sp,
                    )
                )

            systems.append(
                SystemLayout(
                    system_index=sys_idx,
                    measure_start=m_start,
                    measure_end=m_end,
                    x_left=x_left,
                    x_right=x_right,
                    bar_x_bounds=bar_bounds,
                    staves=staves,
                    is_blank_prerule=False,
                )
            )
            sys_idx += 1
            y_cursor += system_h + inter_system_gap
        else:
            # Remaining vertical space: optionally draw blank pre-ruled manuscript staves
            if prerule_prob > 0.0 and rng.random() < prerule_prob:
                staves = [
                    StaffLayout(
                        part_index=0,
                        part_name="blank",
                        brace_group=None,
                        x_left=x_left,
                        x_right=x_right,
                        y_center=y_cursor,
                        sp=sp,
                    )
                ]
                systems.append(
                    SystemLayout(
                        system_index=sys_idx,
                        measure_start=0,
                        measure_end=0,
                        x_left=x_left,
                        x_right=x_right,
                        bar_x_bounds=[],
                        staves=staves,
                        is_blank_prerule=True,
                    )
                )
                sys_idx += 1
                y_cursor += staff_h + inter_system_gap
            else:
                break

    return systems


# ---------------------------------------------------------------------------
# Adaptive Helpers for Engine Calls (Resilient to Minor Signature Variations)
# ---------------------------------------------------------------------------


def _eval_cubic_bezier_y(ctrl_pts: Sequence[Sequence[float]], x: float) -> float:
    """Evaluate y(x) on a horizontal-monotonic cubic Bezier curve."""
    if not ctrl_pts:
        return 0.0
    x0 = float(ctrl_pts[0][0])
    x3 = float(ctrl_pts[-1][0])
    if abs(x3 - x0) < 1e-6:
        return float(ctrl_pts[0][1])
    t = max(0.0, min(1.0, (x - x0) / (x3 - x0)))
    if len(ctrl_pts) == 4:
        mt = 1.0 - t
        return (
            (mt ** 3) * float(ctrl_pts[0][1])
            + 3.0 * (mt ** 2) * t * float(ctrl_pts[1][1])
            + 3.0 * mt * (t ** 2) * float(ctrl_pts[2][1])
            + (t ** 3) * float(ctrl_pts[3][1])
        )
    # Linear interpolation fallback
    return (1.0 - t) * float(ctrl_pts[0][1]) + t * float(ctrl_pts[-1][1])


def _get_staff_curve(
    layout_mod: Any,
    x_left: float,
    x_right: float,
    y_center: float,
    sp: float,
    style: Dict[str, Any],
    np_rng: np.random.Generator,
) -> Tuple[Callable[[Any, int], Any], List[List[float]], float]:
    """Build curved staff y(x, line_idx) function, 4 Bezier control points, and effective sp."""
    fn = getattr(layout_mod, "build_curved_staff_y", None)
    if fn is not None:
        res = fn(
            x0=x_left,
            x1=x_right,
            y_center=y_center,
            sp=sp,
            style=style,
            rng=np_rng,
        )
        if isinstance(res, tuple) and len(res) >= 3:
            y_fn, ctrl, sp_out = res[0], res[1], float(res[2])
            ctrl_list = [[float(pt[0]), float(pt[1])] for pt in ctrl]
            return y_fn, ctrl_list, sp_out
        if isinstance(res, tuple) and len(res) == 2:
            y_fn_1, ctrl = res[0], res[1]
            ctrl_list = [[float(pt[0]), float(pt[1])] for pt in ctrl]
            return (lambda x, li=0, f=y_fn_1, s=sp: f(x) + li * s), ctrl_list, sp

    # Deterministic smooth fallback curve
    sag = float(style.get("staff_curve_px", 4.0)) * float(np_rng.uniform(-1.0, 1.0))
    tilt = float(style.get("staff_slope_deg", 0.5)) * 4.0 * float(np_rng.uniform(-1.0, 1.0))
    w = x_right - x_left
    ctrl_list = [
        [x_left, y_center - tilt * 0.5],
        [x_left + w / 3.0, y_center + sag - tilt * 0.15],
        [x_left + 2.0 * w / 3.0, y_center + sag + tilt * 0.15],
        [x_right, y_center + tilt * 0.5],
    ]
    return (lambda x, li=0, c=ctrl_list, s=sp: _eval_cubic_bezier_y(c, float(x)) + li * s), ctrl_list, sp


def _compute_note_xs(
    layout_mod: Any,
    x_start: float,
    x_end: float,
    sp: float,
    events: List[ScoreEvent],
    style: Dict[str, Any],
    np_rng: np.random.Generator,
) -> List[float]:
    """Compute quirky horizontal note positions inside a measure."""
    n = len(events)
    if n == 0:
        return []
    fn = getattr(layout_mod, "compute_quirky_note_positions", None)
    if fn is not None and x_end > x_start + 2.0:
        offsets = [float(ev.beat_offset) for ev in events]
        try:
            xs = fn(
                beat_offsets=offsets,
                bar_x0=x_start,
                bar_x1=x_end,
                sp=sp,
                style=style,
                rng=np_rng,
            )
            if xs is not None and len(xs) == n:
                return [float(x) for x in xs]
        except Exception:
            pass

    # Fallback: proportional beat spacing + handwritten crowding/jitter
    total_beats = max(sum(ev.beats for ev in events), 1e-3)
    pad_l = (x_end - x_start) * 0.08
    pad_r = (x_end - x_start) * 0.08
    usable = max(10.0, (x_end - x_start) - pad_l - pad_r)
    jitter_scale = float(style.get("spacing_jitter", 0.15)) * 12.0
    xs_out: List[float] = []
    acc = 0.0
    for ev in events:
        frac = acc / total_beats
        x = x_start + pad_l + frac * usable + float(np_rng.uniform(-jitter_scale, jitter_scale))
        xs_out.append(max(x_start + 4.0, min(x_end - 4.0, x)))
        acc += ev.beats
    return xs_out


def _call_glyph(fn: Callable[..., Any], **candidates: Any) -> Any:
    """Invoke a glyph/stroke function passing only accepted keyword arguments."""
    sig = inspect.signature(fn)
    params = sig.parameters
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if has_var_kw:
        return fn(**candidates)
    filtered = {k: v for k, v in candidates.items() if k in params}
    return fn(**filtered)


# ---------------------------------------------------------------------------
# Main Handwritten Score Renderer
# ---------------------------------------------------------------------------


def _draw_ledger_lines(
    canvas: Any,
    stroke_mod: Any,
    x: float,
    y_center: float,
    sp: float,
    steps: List[int],
    ink_rgb: Any,
    style: Dict[str, Any],
    rng: Any,
) -> None:
    if not steps:
        return
    min_s = min(steps)
    max_s = max(steps)
    half_w = sp * 0.95
    draw_poly = getattr(stroke_mod, "draw_calligraphic_polyline", None)
    if draw_poly is None:
        return

    ledger_steps: List[int] = []
    if min_s <= -2:
        ledger_steps.extend(range(-2, min_s - 1, -2))
    if max_s >= 10:
        ledger_steps.extend(range(10, max_s + 1, 2))

    w_px = max(1.0, sp * 0.09 * float(style.get("stroke_scale", 1.0)))
    for ls in ledger_steps:
        ly = y_center - (ls - 4) * (sp * 0.5)
        pts = np.asarray(
            [
                [x - half_w, ly + rng.uniform(-0.5, 0.5)],
                [x, ly + rng.uniform(-0.4, 0.4)],
                [x + half_w, ly + rng.uniform(-0.5, 0.5)],
            ],
            dtype=np.float32,
        )
        draw_poly(
            canvas=canvas,
            pts=pts,
            color_rgb=ink_rgb,
            base_width=w_px,
            style=style,
            kind="ledger",
        )


def render_score_handwritten(
    score_path: Union[str, Path],
    style_name: str,
    out_png: Union[str, Path],
    seed: int = 42,
    page_w: int = 1650,
    page_h: int = 2200,
) -> Dict[str, Any]:
    """Render a MusicXML / MuseScore file in the given handwritten style.

    Saves `out_png` and a companion ground-truth JSON at `Path(out_png).with_suffix('.json')`.
    Returns the ground-truth metadata dict.
    """
    import glyph_variants
    import layout_quirks
    import stroke_engine
    import styles

    score = parse_score(score_path)
    style = styles.get_style(style_name) if hasattr(styles, "get_style") else styles.SCRIBE_STYLES[style_name]
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    ink_rgb = style.get("ink_rgb", (0.12, 0.10, 0.08))
    staff_ink_rgb = style.get("staff_ink_rgb", ink_rgb)
    paper_rgb = style.get("paper_rgb", (0.95, 0.92, 0.86))
    stroke_scale = float(style.get("stroke_scale", 1.0))

    # 1. Manuscript paper background (float32 in [0, 1] or uint8 in [0, 255])
    paper_arr = layout_quirks.make_manuscript_paper(page_w, page_h, style, np_rng)
    if isinstance(paper_arr, Image.Image):
        paper_f32 = np.asarray(paper_arr.convert("RGB"), dtype=np.float32) / 255.0
    else:
        paper_f32 = np.asarray(paper_arr, dtype=np.float32)
        if float(np.max(paper_f32)) > 1.0 + 1e-3:
            paper_f32 = paper_f32 / 255.0
        paper_f32 = np.clip(paper_f32, 0.0, 1.0)

    # 2. Initialize StrokeCanvas & load textured paper into float32 buffer
    canvas = stroke_engine.StrokeCanvas(
        width=page_w,
        height=page_h,
        bg_rgb=paper_rgb,
        rng=np_rng,
    )
    if hasattr(canvas, "buf") and canvas.buf.shape == paper_f32.shape:
        canvas.buf[:] = paper_f32

    # 3. Compute page systems & curved staff lines
    systems = layout_score_pages(score, page_w=page_w, page_h=page_h, style=style, rng=rng)
    gt_systems: List[Dict[str, Any]] = []

    for sys_lay in systems:
        sys_gt_staves: List[Dict[str, Any]] = []
        staff_curves: List[Tuple[Callable[[Any, int], Any], List[List[float]], float]] = []

        # Draw 5-line curved staves
        for st_lay in sys_lay.staves:
            y_fn_2, ctrl_pts, sp_eff = _get_staff_curve(
                layout_quirks,
                st_lay.x_left,
                st_lay.x_right,
                st_lay.y_center,
                st_lay.sp,
                style,
                np_rng,
            )
            st_lay.sp = sp_eff
            st_lay.ctrl_pts = ctrl_pts
            staff_curves.append((y_fn_2, ctrl_pts, sp_eff))

            staff_lw = max(1.0, sp_eff * 0.085 * stroke_scale)
            for line_idx in range(-2, 3):  # -2 (top) .. +2 (bottom)
                stroke_engine.draw_staff_line(
                    canvas=canvas,
                    x0=st_lay.x_left,
                    x1=st_lay.x_right,
                    y_fn=(lambda x, fn=y_fn_2, li=line_idx: fn(x, li)),
                    color_rgb=staff_ink_rgb,
                    base_width=staff_lw,
                    style=style,
                )

            sys_gt_staves.append(
                {
                    "part_index": st_lay.part_index,
                    "part_name": st_lay.part_name,
                    "sp": round(float(sp_eff), 3),
                    "x_left": round(float(st_lay.x_left), 2),
                    "x_right": round(float(st_lay.x_right), 2),
                    "y_center": round(float(st_lay.y_center), 2),
                    "ctrl_pts": [[round(c[0], 2), round(c[1], 2)] for c in ctrl_pts],
                    "is_blank_prerule": sys_lay.is_blank_prerule,
                }
            )

        gt_systems.append(
            {
                "system_index": sys_lay.system_index,
                "measure_start": sys_lay.measure_start,
                "measure_end": sys_lay.measure_end,
                "is_blank_prerule": sys_lay.is_blank_prerule,
                "staves": sys_gt_staves,
            }
        )

        if sys_lay.is_blank_prerule:
            continue

        # Draw system-left connecting barline & grand-staff braces
        if len(sys_lay.staves) >= 2:
            top_st = sys_lay.staves[0]
            y_top = float(staff_curves[0][0](sys_lay.x_left, -2))
            y_bot = float(staff_curves[-1][0](sys_lay.x_left, 2))
            glyph_variants.draw_barline(
                canvas=canvas,
                x=sys_lay.x_left,
                y_top=y_top,
                y_bot=y_bot,
                sp=top_st.sp,
                kind="single",
                color_rgb=ink_rgb,
                style=style,
                rng=np_rng,
            )

            # Braces for grouped staves (e.g. Piano RH+LH)
            seen_groups: Dict[int, List[int]] = {}
            for idx, st_lay in enumerate(sys_lay.staves):
                if st_lay.brace_group is not None:
                    seen_groups.setdefault(st_lay.brace_group, []).append(idx)
            for idxs in seen_groups.values():
                if len(idxs) >= 2:
                    i0, i1 = idxs[0], idxs[-1]
                    by0 = float(staff_curves[i0][0](sys_lay.x_left, -2))
                    by1 = float(staff_curves[i1][0](sys_lay.x_left, 2))
                    glyph_variants.draw_brace(
                        canvas=canvas,
                        x=sys_lay.x_left - sys_lay.staves[i0].sp * 0.7,
                        y_top=by0,
                        y_bot=by1,
                        sp=sys_lay.staves[i0].sp,
                        color_rgb=ink_rgb,
                        style=style,
                        rng=np_rng,
                    )

        # Render each staff's measures in this system
        for s_idx, st_lay in enumerate(sys_lay.staves):
            part = score.parts[st_lay.part_index]
            y_fn_2, _, sp = staff_curves[s_idx]

            def y_mid(x_val: float, _fn: Callable[[Any, int], Any] = y_fn_2) -> float:
                return float(_fn(x_val, 0))

            for rel_m, m_idx in enumerate(range(sys_lay.measure_start, sys_lay.measure_end)):
                if m_idx >= len(part.measures):
                    continue
                meas = part.measures[m_idx]
                bx0, bx1 = sys_lay.bar_x_bounds[rel_m]
                cursor_x = bx0 + sp * 0.7

                # Clef at start of system or on clef change
                if rel_m == 0 or meas.clef_change:
                    clef_x = cursor_x + sp * 0.9
                    glyph_variants.draw_clef(
                        canvas=canvas,
                        clef_name=meas.clef,
                        cx=clef_x,
                        cy=y_mid(clef_x),
                        sp=sp,
                        color_rgb=ink_rgb,
                        style=style,
                        rng=np_rng,
                    )
                    cursor_x += sp * 2.5

                # Key signature at start of system or on key change
                if (rel_m == 0 or meas.key_change) and meas.key_fifths != 0:
                    acc_kind = "sharp" if meas.key_fifths > 0 else "flat"
                    steps_tbl = (
                        SHARP_STEPS_BY_CLEF.get(meas.clef, SHARP_STEPS_BY_CLEF["G-2"])
                        if meas.key_fifths > 0
                        else FLAT_STEPS_BY_CLEF.get(meas.clef, FLAT_STEPS_BY_CLEF["G-2"])
                    )
                    for k_i in range(abs(meas.key_fifths)):
                        kx = cursor_x + k_i * sp * 0.85
                        ky = y_mid(kx) - (steps_tbl[k_i] - 4) * (sp * 0.5)
                        glyph_variants.draw_accidental(
                            canvas=canvas,
                            acc=acc_kind,
                            cx=kx,
                            cy=ky,
                            sp=sp,
                            color_rgb=ink_rgb,
                            style=style,
                            rng=np_rng,
                        )
                    cursor_x += abs(meas.key_fifths) * sp * 0.85 + sp * 0.7

                # Time signature on measure 0 or on time change
                if m_idx == 0 or meas.time_change:
                    tx = cursor_x + sp * 0.8
                    glyph_variants.draw_time_sig(
                        canvas=canvas,
                        num=meas.time_sig[0],
                        den=meas.time_sig[1],
                        cx=tx,
                        cy=y_mid(tx),
                        sp=sp,
                        color_rgb=ink_rgb,
                        style=style,
                        rng=np_rng,
                    )
                    cursor_x += sp * 2.1

                # Compute horizontal positions for events
                ev_xs = _compute_note_xs(
                    layout_quirks, cursor_x, bx1 - sp * 0.5, sp, meas.events, style, np_rng
                )

                # Pre-group beamed notes so stems share a consistent direction & beam line
                beam_group_dir: Dict[int, int] = {}
                for ev in meas.events:
                    if isinstance(ev, NoteEvent) and ev.beam_group is not None:
                        beam_group_dir.setdefault(ev.beam_group, 0)
                for bg_id in list(beam_group_dir.keys()):
                    steps_in_bg = [
                        s
                        for ev in meas.events
                        if isinstance(ev, NoteEvent) and ev.beam_group == bg_id
                        for s in ev.staff_steps
                    ]
                    avg_s = sum(steps_in_bg) / max(1, len(steps_in_bg))
                    beam_group_dir[bg_id] = -1 if avg_s >= 4.0 else 1

                # Track beamed notes: bg_id -> list of (stem_x, far_y, tip_y, sdir, dur_kind)
                beam_groups: Dict[int, List[Tuple[float, float, float, int, str]]] = {}

                for ev, ex in zip(meas.events, ev_xs):
                    yc = y_mid(ex)
                    if isinstance(ev, RestEvent):
                        glyph_variants.draw_rest(
                            canvas=canvas,
                            rest_type=ev.duration_kind,
                            cx=ex,
                            cy=yc,
                            sp=sp,
                            color_rgb=ink_rgb,
                            style=style,
                            rng=np_rng,
                        )
                        continue

                    # NoteEvent / Chord
                    _draw_ledger_lines(canvas, stroke_engine, ex, yc, sp, ev.staff_steps, ink_rgb, style, rng)

                    if ev.beam_group is not None and ev.beam_group in beam_group_dir:
                        stem_dir = beam_group_dir[ev.beam_group]
                    elif ev.stem_dir is not None:
                        stem_dir = ev.stem_dir
                    else:
                        avg_step = sum(ev.staff_steps) / len(ev.staff_steps)
                        stem_dir = -1 if avg_step >= 4.0 else 1

                    # Draw noteheads, accidentals, and augmentation dots
                    ys_note: List[float] = []
                    for st_step, acc_kind in zip(ev.staff_steps, ev.accidentals):
                        ny = yc - (st_step - 4) * (sp * 0.5)
                        ys_note.append(ny)

                        if acc_kind:
                            glyph_variants.draw_accidental(
                                canvas=canvas,
                                acc=acc_kind,
                                cx=ex - sp * 1.25,
                                cy=ny,
                                sp=sp,
                                color_rgb=ink_rgb,
                                style=style,
                                rng=np_rng,
                            )

                        glyph_variants.draw_notehead(
                            canvas=canvas,
                            duration_kind=ev.duration_kind,
                            cx=ex,
                            cy=ny,
                            sp=sp,
                            color_rgb=ink_rgb,
                            style=style,
                            rng=np_rng,
                        )

                        # Augmentation dots
                        if ev.dots > 0:
                            dot_y = ny - (sp * 0.28 if st_step % 2 == 0 else 0.0)
                            for d_i in range(ev.dots):
                                dx = ex + sp * (0.95 + d_i * 0.45)
                                stroke_engine.draw_filled_blob(
                                    canvas=canvas,
                                    cx=dx,
                                    cy=dot_y,
                                    rx=max(1.5, sp * 0.14),
                                    ry=max(1.5, sp * 0.14),
                                    angle_deg=0.0,
                                    color_rgb=ink_rgb,
                                    style=style,
                                    hollow=False,
                                )

                    # Stem & Flag (if not whole note)
                    if ev.duration_kind != "whole":
                        anchor_y = min(ys_note) if stem_dir == 1 else max(ys_note)
                        far_y = max(ys_note) if stem_dir == 1 else min(ys_note)
                        stem_len = max(3.3 * sp, abs(far_y - anchor_y) + 2.8 * sp)
                        stem_x = ex + (sp * 0.52 if stem_dir == 1 else -sp * 0.52)
                        tip_y = anchor_y - stem_dir * stem_len

                        if ev.beam_group is not None:
                            beam_groups.setdefault(ev.beam_group, []).append(
                                (stem_x, far_y, tip_y, stem_dir, ev.duration_kind)
                            )
                        else:
                            n_flags = {"eighth": 1, "16th": 2, "32nd": 3}.get(ev.duration_kind, 0)
                            glyph_variants.draw_stem_and_flag(
                                canvas=canvas,
                                x=stem_x,
                                y0=far_y,
                                y1=tip_y,
                                sp=sp,
                                flags=n_flags,
                                stem_up=(stem_dir == 1),
                                color_rgb=ink_rgb,
                                style=style,
                                rng=np_rng,
                            )

                # Draw stems + beams for grouped 8th/16th/32nd notes so stems meet the beam cleanly
                for grp_items in beam_groups.values():
                    if len(grp_items) == 1:
                        sx, fy, ty, sdir, dk = grp_items[0]
                        n_flags = {"eighth": 1, "16th": 2, "32nd": 3}.get(dk, 1)
                        glyph_variants.draw_stem_and_flag(
                            canvas=canvas,
                            x=sx,
                            y0=fy,
                            y1=ty,
                            sp=sp,
                            flags=n_flags,
                            stem_up=(sdir == 1),
                            color_rgb=ink_rgb,
                            style=style,
                            rng=np_rng,
                        )
                        continue

                    x0, _, y0_raw, sdir, _ = grp_items[0]
                    x1, _, y1_raw, _, _ = grp_items[-1]
                    max_dy = min(sp * 1.4, abs(x1 - x0) * 0.22)
                    dy = max(-max_dy, min(max_dy, y1_raw - y0_raw))
                    # Shift beam outward if any interior note would be too close
                    y0 = y0_raw
                    y1 = y0 + dy
                    dx_span = max(1e-3, x1 - x0)
                    for sx, fy, ty, _, _ in grp_items:
                        t = (sx - x0) / dx_span
                        by = y0 + t * (y1 - y0)
                        if sdir == 1 and by > ty:
                            shift = ty - by
                            y0 += shift
                            y1 += shift
                        elif sdir == -1 and by < ty:
                            shift = ty - by
                            y0 += shift
                            y1 += shift

                    for sx, fy, _, _, _ in grp_items:
                        t = (sx - x0) / dx_span
                        by = y0 + t * (y1 - y0)
                        glyph_variants.draw_stem_and_flag(
                            canvas=canvas,
                            x=sx,
                            y0=fy,
                            y1=by,
                            sp=sp,
                            flags=0,
                            stem_up=(sdir == 1),
                            color_rgb=ink_rgb,
                            style=style,
                            rng=np_rng,
                        )

                    n_beams = max(
                        {"eighth": 1, "16th": 2, "32nd": 3}.get(it[4], 1) for it in grp_items
                    )
                    for b_i in range(n_beams):
                        b_off = b_i * sp * 0.62 * sdir
                        glyph_variants.draw_beam(
                            canvas=canvas,
                            x0=x0,
                            y0=y0 + b_off,
                            x1=x1,
                            y1=y1 + b_off,
                            sp=sp,
                            color_rgb=ink_rgb,
                            style=style,
                            rng=np_rng,
                        )

                # Measure right barline
                bar_y0 = float(y_fn_2(bx1, -2))
                bar_y1 = float(y_fn_2(bx1, 2))
                glyph_variants.draw_barline(
                    canvas=canvas,
                    x=bx1,
                    y_top=bar_y0,
                    y_bot=bar_y1,
                    sp=sp,
                    kind=meas.barline,
                    color_rgb=ink_rgb,
                    style=style,
                    rng=np_rng,
                )

    # 4. Composite & save output PNG + companion GT JSON
    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_img = canvas.to_pil()
    final_img.save(out_path)

    gt_payload: Dict[str, Any] = {
        "score_title": score.title,
        "source_path": str(score_path),
        "style": style_name,
        "seed": seed,
        "page_w": page_w,
        "page_h": page_h,
        "systems": gt_systems,
    }
    gt_path = out_path.with_suffix(".json")
    gt_path.write_text(json.dumps(gt_payload, indent=2), encoding="utf-8")
    return gt_payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render a MusicXML or MuseScore (.mscx/.mscz) file in handwritten styles."
    )
    parser.add_argument("--input", required=True, help="Path to .musicxml, .mxl, .mscx, or .mscz")
    parser.add_argument(
        "--styles",
        default="beethoven_hurried_autograph,debussy_pencil_sketch,student_ballpoint_blue",
        help="Comma-separated style names or 'all' for all 20 styles",
    )
    parser.add_argument(
        "--out-dir",
        default="tmp-train/hw-engine/mvp_samples",
        help="Directory to write rendered PNGs and GT JSON files",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed")
    parser.add_argument("--page-w", type=int, default=1650, help="Output page width in pixels")
    parser.add_argument("--page-h", type=int, default=2200, help="Output page height in pixels")
    args = parser.parse_args(argv)

    import styles as styles_mod

    if args.styles.strip().lower() == "all":
        style_names = list(styles_mod.SCRIBE_STYLES.keys())
    else:
        style_names = [s.strip() for s in args.styles.split(",") if s.strip()]

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, sname in enumerate(style_names):
        out_png = out_dir / f"{in_path.stem}__{sname}.png"
        gt = render_score_handwritten(
            score_path=in_path,
            style_name=sname,
            out_png=out_png,
            seed=args.seed + idx,
            page_w=args.page_w,
            page_h=args.page_h,
        )
        print(
            f"[ok] {sname}: {out_png} "
            f"({len(gt['systems'])} systems, gt={out_png.with_suffix('.json').name})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
