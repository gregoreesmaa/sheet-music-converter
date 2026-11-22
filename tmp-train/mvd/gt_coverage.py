"""GT coverage extraction & audit for MusicXML -> deterministic JSON sidecar.

Addresses missing ground-truth coverage for:
  1. Dynamics & hairpins: full MusicXML <dynamics> (including ppp, fff, fp,
     sf, sfp, sffz, rf, rfz, fz, other-dynamics) and <wedge> (crescendo,
     diminuendo, stop, continue), preserving multiple events per measure/onset.
  2. Lyrics & text: measure/beat-aligned <lyric> syllables (verse, syllabic,
     text, extend), <direction-type><words>, <rehearsal>, <metronome>, and
     <harmony> chord symbols.
  3. Drum / tab notation (Gate-D opt-in, OFF by default):
     - Always detects percussion/drum/slash/tab staves and records
       ``has_percussion_or_tab`` + ``excluded_staves`` so Gate A-C training
       can enforce todo.md Step 3's hard boundary ("NO drum-kit/slash-note
       staves until Gate D").
     - Only populates ``percussion``, ``tab``, ``slash_spans``, and
       ``staff_tunings`` when ``include_percussion_tab=True``
       (CLI ``--include-percussion-tab``).

Dependencies: Python standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

PathLike = Union[str, Path]

SCHEMA_VERSION = 1

# Closed dynamic tokens currently supported inline by bekern+ (data.py VOCAB).
BEKERN_DYN_MAP: Dict[str, str] = {
    "pp": "dyn-pp",
    "p": "dyn-p",
    "mp": "dyn-mp",
    "mf": "dyn-mf",
    "f": "dyn-f",
    "ff": "dyn-ff",
    "sfz": "dyn-sfz",
}

BEKERN_WEDGE_MAP: Dict[str, str] = {
    "crescendo": "hairpin-cresc-start",
    "diminuendo": "hairpin-dim-start",
    "stop": "hairpin-stop",
}

KNOWN_NOTEHEADS = frozenset(
    ("x", "diamond", "slash", "triangle", "circle-x", "normal")
)

_DURATION_TABLE: Tuple[Tuple[str, float, int], ...] = (
    ("128th", 0.03125, 0),
    ("64th", 0.0625, 0),
    ("64th", 0.09375, 1),
    ("32nd", 0.125, 0),
    ("32nd", 0.1875, 1),
    ("32nd", 0.21875, 2),
    ("16th", 0.25, 0),
    ("16th", 0.375, 1),
    ("16th", 0.4375, 2),
    ("eighth", 0.5, 0),
    ("eighth", 0.75, 1),
    ("eighth", 0.875, 2),
    ("quarter", 1.0, 0),
    ("quarter", 1.5, 1),
    ("quarter", 1.75, 2),
    ("half", 2.0, 0),
    ("half", 3.0, 1),
    ("half", 3.5, 2),
    ("whole", 4.0, 0),
    ("whole", 6.0, 1),
    ("whole", 7.0, 2),
    ("breve", 8.0, 0),
)


def _strip_ns(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def _child_text(elem: ET.Element, name: str) -> Optional[str]:
    for ch in elem:
        if _strip_ns(ch.tag) == name:
            txt = "".join(ch.itertext()).strip()
            return txt if txt else None
    return None


def _round3(val: float) -> float:
    return round(val, 3)


def _parse_int(text: Optional[str], default: Optional[int] = None) -> Optional[int]:
    if text is None:
        return default
    try:
        return int(text.strip())
    except ValueError:
        return default


def _duration_token(dur_beats: float, tol: float = 0.005) -> Optional[Tuple[str, int]]:
    best: Optional[Tuple[str, int]] = None
    best_dist = float("inf")
    for name, beats, dots in _DURATION_TABLE:
        dist = abs(beats - dur_beats)
        if dist < best_dist:
            best_dist = dist
            best = (name, dots)
    if best is not None and best_dist <= tol:
        return best
    return None


def _grace_beats(note_type: Optional[str]) -> float:
    mapping = {
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
    return mapping.get((note_type or "").strip(), 0.5)


def _format_harmony(elem: ET.Element) -> Optional[str]:
    root_el = None
    kind_text = None
    bass_el = None
    for ch in elem:
        t = _strip_ns(ch.tag)
        if t == "root":
            root_el = ch
        elif t == "kind":
            kind_text = (ch.attrib.get("text") or (ch.text or "")).strip()
        elif t == "bass":
            bass_el = ch
    if root_el is None:
        return None
    step = _child_text(root_el, "root-step")
    if not step:
        return None
    alter = _parse_int(_child_text(root_el, "root-alter"), 0) or 0
    acc = {1: "#", 2: "##", -1: "b", -2: "bb"}.get(alter, "")
    out = f"{step}{acc}"
    if kind_text and kind_text != "major":
        out += f":{kind_text}"
    if bass_el is not None:
        bstep = _child_text(bass_el, "bass-step")
        if bstep:
            balter = _parse_int(_child_text(bass_el, "bass-alter"), 0) or 0
            bacc = {1: "#", 2: "##", -1: "b", -2: "bb"}.get(balter, "")
            out += f"/{bstep}{bacc}"
    return out


def extract_gt_sidecar(
    xml_text: str,
    *,
    include_percussion_tab: bool = False,
    measures: Optional[Tuple[int, int]] = None,
) -> Dict:
    """Extract deterministic GT coverage sidecar from MusicXML Partwise text.

    Args:
        xml_text: Raw MusicXML Partwise 3.1/4.0 XML string.
        include_percussion_tab: Opt-in Gate-D flag (default ``False``). When
            ``False``, percussion/tab staves are still detected and reported in
            ``has_percussion_or_tab`` and ``excluded_staves``, but
            ``percussion``, ``tab``, ``slash_spans``, and ``staff_tunings``
            remain empty.
        measures: Optional inclusive 1-based ``(start, end)`` measure filter.

    Returns:
        Deterministic JSON-serializable sidecar dict (``schema_version == 1``).
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ValueError(f"invalid MusicXML: {exc}") from None

    if _strip_ns(root.tag) != "score-partwise":
        raise ValueError(f"expected score-partwise root, got {root.tag!r}")

    if measures is not None:
        m_start, m_end = measures
        if m_start < 1 or m_end < m_start:
            raise ValueError(f"invalid measure range {measures!r}")
    else:
        m_start, m_end = None, None

    # Parse <part-list> score-instrument names (used for drum kit mapping).
    inst_names: Dict[str, str] = {}
    for el in root.iter():
        if _strip_ns(el.tag) == "score-instrument":
            iid = el.attrib.get("id")
            iname = _child_text(el, "instrument-name")
            if iid and iname:
                inst_names[iid] = iname

    dynamics: List[Dict] = []
    hairpins: List[Dict] = []
    lyrics: List[Dict] = []
    text_directions: List[Dict] = []
    percussion: List[Dict] = []
    tab: List[Dict] = []
    slash_spans: List[Dict] = []
    staff_tunings: List[Dict] = []
    excluded_staves: set[int] = set()

    staff_base = 0
    for part_el in root:
        if _strip_ns(part_el.tag) != "part":
            continue
        divisions = 1
        part_staves = 1
        measure_els = [ch for ch in part_el if _strip_ns(ch.tag) == "measure"]

        # Pre-scan part_staves for stable global staff indexing (mirrors
        # musicxml_import.rs).
        probe_staff = 1
        for m_el in measure_els:
            for item in m_el:
                tag = _strip_ns(item.tag)
                if tag == "attributes":
                    st = _parse_int(_child_text(item, "staves"))
                    if st is not None and st > 0:
                        part_staves = max(part_staves, st)
                elif tag == "note":
                    st = _parse_int(_child_text(item, "staff"))
                    local = max(st if st is not None else probe_staff, 1)
                    probe_staff = local
                    part_staves = max(part_staves, local)

        for ord0, m_el in enumerate(measure_els):
            measure_num = ord0 + 1
            in_range = (
                m_start is None or (m_start <= measure_num <= m_end)  # type: ignore[operator]
            )
            cursor = 0
            last_onset = 0
            last_staff = 1

            for item in m_el:
                tag = _strip_ns(item.tag)

                if tag == "attributes":
                    d = _parse_int(_child_text(item, "divisions"))
                    if d is not None and d > 0:
                        divisions = d
                    for ch in item:
                        ctag = _strip_ns(ch.tag)
                        if ctag == "clef":
                            cnum = max(_parse_int(ch.attrib.get("number"), 1) or 1, 1)
                            sign = (_child_text(ch, "sign") or "").strip()
                            if sign.lower() == "percussion" or sign.upper() == "TAB":
                                excluded_staves.add(staff_base + cnum - 1)
                        elif ctag == "measure-style":
                            for mch in ch:
                                if _strip_ns(mch.tag) == "slash":
                                    stype = (mch.attrib.get("type") or "start").strip()
                                    gstaff = staff_base + last_staff - 1
                                    excluded_staves.add(gstaff)
                                    if include_percussion_tab and in_range:
                                        slash_spans.append({
                                            "measure": measure_num,
                                            "onset": _round3(cursor / divisions),
                                            "staff": gstaff,
                                            "span_type": stype,
                                        })
                        elif ctag == "staff-details":
                            snum = max(_parse_int(ch.attrib.get("number"), 1) or 1, 1)
                            tunings = []
                            for sch in ch:
                                if _strip_ns(sch.tag) == "staff-tuning":
                                    line = _parse_int(sch.attrib.get("line"), 1) or 1
                                    tstep = (_child_text(sch, "tuning-step") or "E").upper()
                                    talter = _parse_int(_child_text(sch, "tuning-alter"), 0) or 0
                                    toct = _parse_int(_child_text(sch, "tuning-octave"), 2) or 2
                                    tunings.append({
                                        "line": line,
                                        "step": tstep,
                                        "alter": talter,
                                        "octave": toct,
                                    })
                            if tunings:
                                gstaff = staff_base + snum - 1
                                excluded_staves.add(gstaff)
                                if include_percussion_tab:
                                    staff_tunings.append({
                                        "staff": gstaff,
                                        "measure": measure_num,
                                        "tunings": tunings,
                                    })

                elif tag == "backup":
                    d = _parse_int(_child_text(item, "duration"), 0) or 0
                    cursor -= d

                elif tag == "forward":
                    d = _parse_int(_child_text(item, "duration"), 0) or 0
                    cursor += d

                elif tag == "direction" and in_range:
                    dir_staff = _parse_int(_child_text(item, "staff"))
                    local_staff = max(dir_staff if dir_staff is not None else last_staff, 1)
                    gstaff = staff_base + local_staff - 1
                    dir_voice = _parse_int(_child_text(item, "voice"), 1) or 1
                    onset = _round3(cursor / divisions)
                    for dt in item:
                        if _strip_ns(dt.tag) != "direction-type":
                            continue
                        for dch in dt:
                            dtag = _strip_ns(dch.tag)
                            if dtag == "dynamics":
                                for mark_el in dch:
                                    mtag = _strip_ns(mark_el.tag)
                                    if mtag == "other-dynamics":
                                        mark = "".join(mark_el.itertext()).strip() or "other"
                                    else:
                                        mark = mtag
                                    dynamics.append({
                                        "measure": measure_num,
                                        "onset": onset,
                                        "staff": gstaff,
                                        "voice": dir_voice,
                                        "mark": mark,
                                        "bekern_token": BEKERN_DYN_MAP.get(mark),
                                    })
                            elif dtag == "wedge":
                                wtype = (dch.attrib.get("type") or "").strip()
                                if wtype in ("crescendo", "diminuendo", "stop", "continue"):
                                    wnum = max(_parse_int(dch.attrib.get("number"), 1) or 1, 1)
                                    hairpins.append({
                                        "measure": measure_num,
                                        "onset": onset,
                                        "staff": gstaff,
                                        "voice": dir_voice,
                                        "wedge_type": wtype,
                                        "number": wnum,
                                        "bekern_token": BEKERN_WEDGE_MAP.get(wtype),
                                    })
                            elif dtag == "words":
                                txt = "".join(dch.itertext()).strip()
                                if txt:
                                    text_directions.append({
                                        "measure": measure_num,
                                        "onset": onset,
                                        "staff": gstaff,
                                        "voice": dir_voice,
                                        "kind": "words",
                                        "text": txt,
                                    })
                            elif dtag == "rehearsal":
                                txt = "".join(dch.itertext()).strip()
                                if txt:
                                    text_directions.append({
                                        "measure": measure_num,
                                        "onset": onset,
                                        "staff": gstaff,
                                        "voice": dir_voice,
                                        "kind": "rehearsal",
                                        "text": txt,
                                    })
                            elif dtag == "metronome":
                                bunit = _child_text(dch, "beat-unit") or ""
                                pm = _child_text(dch, "per-minute") or ""
                                txt = f"{bunit}={pm}".strip("=")
                                if txt:
                                    text_directions.append({
                                        "measure": measure_num,
                                        "onset": onset,
                                        "staff": gstaff,
                                        "voice": dir_voice,
                                        "kind": "metronome",
                                        "text": txt,
                                    })

                elif tag == "harmony" and in_range:
                    h_staff = _parse_int(_child_text(item, "staff"))
                    local_staff = max(h_staff if h_staff is not None else last_staff, 1)
                    gstaff = staff_base + local_staff - 1
                    htxt = _format_harmony(item)
                    if htxt:
                        text_directions.append({
                            "measure": measure_num,
                            "onset": _round3(cursor / divisions),
                            "staff": gstaff,
                            "voice": 1,
                            "kind": "harmony",
                            "text": htxt,
                        })

                elif tag == "note":
                    is_chord = any(_strip_ns(c.tag) == "chord" for c in item)
                    is_grace = any(_strip_ns(c.tag) == "grace" for c in item)
                    is_rest = any(_strip_ns(c.tag) == "rest" for c in item)
                    dur = _parse_int(_child_text(item, "duration"), 0) or 0
                    if is_grace:
                        dur_beats = _grace_beats(_child_text(item, "type"))
                    else:
                        dur_beats = dur / divisions
                    onset_ticks = last_onset if is_chord else cursor
                    onset = _round3(onset_ticks / divisions)
                    n_staff = _parse_int(_child_text(item, "staff"))
                    local_staff = max(n_staff if n_staff is not None else last_staff, 1)
                    last_staff = local_staff
                    gstaff = staff_base + local_staff - 1
                    voice = _parse_int(_child_text(item, "voice"), 1) or 1

                    # Lyrics on this note (always extracted regardless of Gate-D flag).
                    if in_range:
                        for ch in item:
                            if _strip_ns(ch.tag) == "lyric":
                                verse = (ch.attrib.get("number") or "1").strip()
                                syllabic = _child_text(ch, "syllabic")
                                text = _child_text(ch, "text")
                                extend = any(_strip_ns(lc.tag) == "extend" for lc in ch)
                                if text is not None or extend:
                                    lyrics.append({
                                        "measure": measure_num,
                                        "onset": onset,
                                        "staff": gstaff,
                                        "voice": voice,
                                        "verse": verse,
                                        "syllabic": syllabic,
                                        "text": text or "",
                                        "extend": extend,
                                    })

                    # Detect unpitched percussion notes.
                    unpitched_el = None
                    pitch_el = None
                    notehead_text = None
                    inst_id = None
                    notations_el = None
                    for ch in item:
                        ctag = _strip_ns(ch.tag)
                        if ctag == "unpitched":
                            unpitched_el = ch
                        elif ctag == "pitch":
                            pitch_el = ch
                        elif ctag == "notehead":
                            notehead_text = "".join(ch.itertext()).strip().lower() or None
                        elif ctag == "instrument":
                            inst_id = ch.attrib.get("id")
                        elif ctag == "notations":
                            notations_el = ch

                    if unpitched_el is not None:
                        excluded_staves.add(gstaff)
                        if include_percussion_tab and in_range:
                            dstep = (_child_text(unpitched_el, "display-step") or "C").upper()
                            doct = _parse_int(_child_text(unpitched_el, "display-octave"), 4) or 4
                            percussion.append({
                                "measure": measure_num,
                                "onset": onset,
                                "dur_beats": _round3(dur_beats),
                                "staff": gstaff,
                                "voice": voice,
                                "display_step": dstep,
                                "display_octave": doct,
                                "notehead": notehead_text,
                                "instrument_id": inst_id,
                                "instrument_name": inst_names.get(inst_id) if inst_id else None,
                                "chord": is_chord,
                                "grace": is_grace,
                            })

                    # Detect tab <technical><string><fret>.
                    if notations_el is not None:
                        for nch in notations_el:
                            if _strip_ns(nch.tag) == "technical":
                                t_str = _parse_int(_child_text(nch, "string"))
                                t_fret = _parse_int(_child_text(nch, "fret"))
                                if t_str is not None and t_fret is not None:
                                    excluded_staves.add(gstaff)
                                    if include_percussion_tab and in_range:
                                        pitch_str = None
                                        if pitch_el is not None and not is_rest:
                                            pstep = (_child_text(pitch_el, "step") or "C").upper()
                                            palter = _parse_int(_child_text(pitch_el, "alter"), 0) or 0
                                            poct = _parse_int(_child_text(pitch_el, "octave"), 4) or 4
                                            acc = {1: "#", 2: "x", -1: "b", -2: "bb"}.get(palter, "")
                                            pitch_str = f"{pstep}{acc}{poct}"
                                        tab.append({
                                            "measure": measure_num,
                                            "onset": onset,
                                            "dur_beats": _round3(dur_beats),
                                            "staff": gstaff,
                                            "voice": voice,
                                            "string": t_str,
                                            "fret": t_fret,
                                            "pitch": pitch_str,
                                            "chord": is_chord,
                                            "grace": is_grace,
                                        })

                    if not is_chord:
                        last_onset = cursor
                        if not is_grace:
                            cursor += dur

        staff_base += part_staves

    # Deterministic sorting across all channels.
    dynamics.sort(key=lambda e: (e["measure"], e["onset"], e["staff"], e["voice"], e["mark"]))
    hairpins.sort(key=lambda e: (e["measure"], e["onset"], e["staff"], e["voice"], e["number"], e["wedge_type"]))
    lyrics.sort(key=lambda e: (e["measure"], e["onset"], e["staff"], e["voice"], e["verse"]))
    text_directions.sort(key=lambda e: (e["measure"], e["onset"], e["staff"], e["voice"], e["kind"], e["text"]))
    percussion.sort(key=lambda e: (e["measure"], e["onset"], e["staff"], e["voice"], e["display_octave"], e["display_step"]))
    tab.sort(key=lambda e: (e["measure"], e["onset"], e["staff"], e["voice"], e["string"], e["fret"]))
    slash_spans.sort(key=lambda e: (e["measure"], e["onset"], e["staff"], e["span_type"]))
    staff_tunings.sort(key=lambda e: (e["staff"], e["measure"]))

    return {
        "schema_version": SCHEMA_VERSION,
        "include_percussion_tab": bool(include_percussion_tab),
        "has_percussion_or_tab": bool(excluded_staves),
        "excluded_staves": sorted(excluded_staves),
        "dynamics": dynamics,
        "hairpins": hairpins,
        "lyrics": lyrics,
        "text_directions": text_directions,
        "percussion": percussion,
        "tab": tab,
        "slash_spans": slash_spans,
        "staff_tunings": staff_tunings,
    }


def audit_bekern_coverage(xml_text: str) -> Dict:
    """Audit what bekern+ already covers vs what is dropped in a MusicXML file."""
    full = extract_gt_sidecar(xml_text, include_percussion_tab=True)
    dyn_in = sum(1 for d in full["dynamics"] if d["bekern_token"] is not None)
    dyn_out = sum(1 for d in full["dynamics"] if d["bekern_token"] is None)
    hp_in = sum(1 for h in full["hairpins"] if h["bekern_token"] is not None)
    hp_out = sum(1 for h in full["hairpins"] if h["bekern_token"] is None)
    text_by_kind: Dict[str, int] = {}
    for td in full["text_directions"]:
        text_by_kind[td["kind"]] = text_by_kind.get(td["kind"], 0) + 1
    return {
        "dynamics_in_bekern": dyn_in,
        "dynamics_dropped_by_bekern": dyn_out,
        "hairpins_in_bekern": hp_in,
        "hairpins_dropped_by_bekern": hp_out,
        "lyrics_count": len(full["lyrics"]),
        "text_directions_by_kind": text_by_kind,
        "percussion_events": len(full["percussion"]),
        "tab_events": len(full["tab"]),
        "slash_spans": len(full["slash_spans"]),
        "has_percussion_or_tab": full["has_percussion_or_tab"],
        "excluded_staves": full["excluded_staves"],
    }


def percussion_tab_to_tokens(sidecar: Dict) -> List[str]:
    """Deterministically serialize Gate-D percussion/tab sidecar events to tokens.

    Refuses unless ``sidecar["include_percussion_tab"]`` is True (enforces
    Gate-D opt-in boundary). Output tokens belong to
    ``data.build_vocab(include_percussion_tab=True)``.
    """
    if not sidecar.get("include_percussion_tab", False):
        raise ValueError(
            "percussion/tab token emission is disabled (Gate-D opt-in required)"
        )
    toks: List[str] = []
    # Group events by measure for deterministic bar-delimited output.
    measures: set[int] = set()
    for ch in ("percussion", "tab", "slash_spans"):
        for ev in sidecar.get(ch, []):
            measures.add(int(ev["measure"]))
    if not measures:
        return toks

    has_tab = bool(sidecar.get("tab"))
    has_perc = bool(sidecar.get("percussion"))
    if has_perc:
        toks.extend(["clef-p-2", "key+0", "time-4-4"])
    if has_tab:
        toks.extend(["clef-TAB-5", "key+0", "time-4-4"])

    for m in sorted(measures):
        for sl in sidecar.get("slash_spans", []):
            if sl["measure"] == m:
                toks.append("slash-start" if sl["span_type"] == "start" else "slash-stop")
        for ev in sidecar.get("percussion", []):
            if ev["measure"] != m:
                continue
            dt = _duration_token(float(ev["dur_beats"]))
            if dt is None:
                continue
            dur_tok, dots = dt
            if ev.get("grace"):
                toks.append("grace")
            toks.extend(["unpitched", dur_tok, ev["display_step"], str(ev["display_octave"])])
            for _ in range(dots):
                toks.append("dot")
            nh = ev.get("notehead")
            if nh in KNOWN_NOTEHEADS:
                toks.append(f"nh-{nh}")
        for ev in sidecar.get("tab", []):
            if ev["measure"] != m:
                continue
            dt = _duration_token(float(ev["dur_beats"]))
            if dt is None:
                continue
            dur_tok, dots = dt
            s_num = int(ev["string"])
            f_num = int(ev["fret"])
            if not (1 <= s_num <= 6 and 0 <= f_num <= 24):
                continue
            if ev.get("grace"):
                toks.append("grace")
            # Emit note skeleton (pitched if known, else C4 placeholder on TAB staff)
            # followed by string/fret technical tokens.
            pitch = ev.get("pitch") or "C4"
            step = pitch[0]
            oct_ch = pitch[-1]
            toks.extend([dur_tok, step, oct_ch])
            for _ in range(dots):
                toks.append("dot")
            toks.extend([f"str-{s_num}", f"fret-{f_num}"])
        toks.append("bar")
    return toks


def extract_gt_sidecar_file(
    xml_path: PathLike,
    out_path: Optional[PathLike] = None,
    *,
    include_percussion_tab: bool = False,
    measures: Optional[Tuple[int, int]] = None,
) -> Dict:
    """Read MusicXML from ``xml_path`` and optionally write sidecar atomically."""
    xml_text = Path(xml_path).read_text(encoding="utf-8")
    sidecar = extract_gt_sidecar(
        xml_text,
        include_percussion_tab=include_percussion_tab,
        measures=measures,
    )
    if out_path is not None:
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        tmp_p = out_p.with_suffix(out_p.suffix + ".tmp")
        tmp_p.write_text(
            json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp_p, out_p)
    return sidecar


def _parse_measures_arg(spec: Optional[str]) -> Optional[Tuple[int, int]]:
    if spec is None:
        return None
    parts = spec.split("..")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"expected START..END, got {spec!r}")
    try:
        a, b = int(parts[0]), int(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError(f"bad measure range {spec!r}") from None
    if a < 1 or b < a:
        raise argparse.ArgumentTypeError(f"invalid measure range {spec!r}")
    return (a, b)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Extract deterministic GT coverage sidecar from MusicXML."
    )
    ap.add_argument("--input", required=True, type=Path, help="Input .musicxml file")
    ap.add_argument("--out", type=Path, help="Output JSON sidecar path")
    ap.add_argument(
        "--measures",
        default=None,
        help="Optional 1-based inclusive measure span START..END",
    )
    ap.add_argument(
        "--include-percussion-tab",
        action="store_true",
        default=False,
        help=(
            "Opt-in Gate-D extraction of unpitched percussion, slash notation, "
            "and guitar tab events (OFF by default for Gate A-C safety)."
        ),
    )
    ap.add_argument(
        "--audit",
        action="store_true",
        default=False,
        help="Print bekern+ coverage audit JSON to stdout instead of full sidecar.",
    )
    args = ap.parse_args(argv)
    mrange = _parse_measures_arg(args.measures)

    if args.audit:
        xml_text = args.input.read_text(encoding="utf-8")
        rep = audit_bekern_coverage(xml_text)
        print(json.dumps(rep, indent=2))
        return 0

    sidecar = extract_gt_sidecar_file(
        args.input,
        args.out,
        include_percussion_tab=args.include_percussion_tab,
        measures=mrange,
    )
    if args.out is None:
        print(json.dumps(sidecar, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
