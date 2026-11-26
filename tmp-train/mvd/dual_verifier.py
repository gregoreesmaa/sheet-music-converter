#!/usr/bin/env python3
"""Dual-Verified Render-Cycle Verifier (R_sym + R_vis -> R_dual) for OMR.

Combines two complementary label-free verifiable rewards in ``[0, 1]``:

1. ``R_sym`` (Exact Symbolic Process Verifier):
   - Measure-level ``bekern`` syntax via ``grammar.PrefixState(strict=False)``
     across ``<sys>`` boundaries (reusing ``rlvr_grpo._walk_parse_and_meter``).
   - Interior-measure beat conservation (``|vcurs - tsig| < 1e-3``) and
     cross-voice duration agreement.
   - Balanced ties & tuplets (reusing ``rlvr_grpo._check_balanced_ties_and_tuplets``).
   - Token repetition-loop AND ``<sys>``-loop penalty.

2. ``R_vis`` (Style- & Spacing-Invariant Visual-Cycle Verifier):
   - Renders predicted ``bekern`` -> MusicXML (via Rust ``assembly-serialize``
     or lightweight pure-Python MusicXML fallback) -> SVG via ``verovio`` ->
     grayscale ink image via ``resvg_py`` (or ``cairosvg`` fallback).
   - Extracts illumination- and contrast-normalized ink maps (invariant to
     paper yellowing, desk lighting, and brown iron-gall vs black engraved ink).
   - Computes:
     (a) **DTW-aligned onset-column Earth Mover's Distance (EMD):** subtracts
         per-row staff-line baseline to isolate noteheads/stems/barlines,
         builds vertical pitch-band histograms per horizontal onset column,
         and aligns real vs re-rendered columns via Sakoe-Chiba banded DTW
         with 1D Wasserstein-1 (EMD) column cost — invariant to scribal
         horizontal stretch and engraving spacing differences.
     (b) **Vertical pitch-density profile correlation & 1D EMD:** compares
         normalized horizontal-projection profiles to verify system count,
         vertical layout, and staff pitch-density envelope.

3. Combined Dual Score:
       ``R_dual = R_sym * (0.2 + 0.8 * R_vis)``
   Prevents empty-bar or hallucinated-valid-music reward hacking while
   strongly rewarding visually grounded symbolic transcriptions.
"""

from __future__ import annotations

from dataclasses import dataclass
import io
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from xml.sax.saxutils import escape as xml_escape

import numpy as np
from PIL import Image, ImageFilter

_THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _THIS_DIR.parent.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import eval_uni_suite31 as eval_s31  # noqa: E402
import grammar  # noqa: E402
import rlvr_grpo  # noqa: E402
import uni_omr  # noqa: E402

try:
    import verovio  # type: ignore
except ImportError:  # pragma: no cover
    verovio = None  # type: ignore

try:
    import resvg_py  # type: ignore
except ImportError:  # pragma: no cover
    resvg_py = None  # type: ignore

try:
    import cairosvg  # type: ignore
except ImportError:  # pragma: no cover
    cairosvg = None  # type: ignore

try:
    from scipy.ndimage import uniform_filter as _uniform_filter  # type: ignore
except ImportError:  # pragma: no cover
    _uniform_filter = None  # type: ignore


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DUAL_SYM_FLOOR: float = 0.20
DUAL_VIS_WEIGHT: float = 0.80

W_SYM_PARSE: float = 0.30
W_SYM_METER: float = 0.55
W_SYM_BALANCE: float = 0.15
SYM_LOOP_PENALTY: float = 0.25

W_VIS_DTW_EMD: float = 0.50
W_VIS_VERT_CORR: float = 0.35
W_VIS_VERT_EMD: float = 0.15

DEFAULT_ONSET_COLS: int = 64
DEFAULT_PITCH_BANDS: int = 12
DEFAULT_VERT_BINS: int = 128
DEFAULT_DTW_BAND: int = 16
DEFAULT_DTW_ALPHA: float = 3.2
DEFAULT_VERT_EMD_ALPHA: float = 6.0

_DUR_DIVISIONS: Dict[str, Tuple[int, str]] = {
    "long": (384, "long"),
    "breve": (192, "breve"),
    "whole": (96, "whole"),
    "half": (48, "half"),
    "quarter": (24, "quarter"),
    "eighth": (12, "eighth"),
    "16th": (6, "16th"),
    "32nd": (3, "32nd"),
    "64th": (2, "64th"),
    "128th": (1, "128th"),
}

_CLEF_XML: Dict[str, Tuple[str, int]] = {
    "clef-G2": ("G", 2),
    "clef-F4": ("F", 4),
    "clef-C3": ("C", 3),
    "clef-C4": ("C", 4),
    "clef-C1": ("C", 1),
    "clef-C2": ("C", 2),
    "clef-G1": ("G", 1),
    "clef-F3": ("F", 3),
}


# ---------------------------------------------------------------------------
# 1. Exact Symbolic Process Verifier (R_sym)
# ---------------------------------------------------------------------------

def detect_sys_loop(tokens: Sequence[str], max_systems: int = 18) -> bool:
    """Detect degenerate ``<sys>``-level repetition loops or excessive system counts.

    Triggers if:
      - Total systems exceed ``max_systems`` (runaway system hallucination).
      - Any empty system exists (consecutive ``<sys> <sys>`` or leading/trailing ``<sys>``).
      - ``>= 3`` consecutive systems have identical token sequences (with ``>= 4`` tokens).
      - ``>= 4`` systems on the page are exact duplicates of each other (when total systems >= 5).
    """
    if not tokens:
        return False

    systems: List[Tuple[str, ...]] = []
    cur: List[str] = []
    for tok in tokens:
        if tok == uni_omr.SYS_TOKEN:
            if not cur:
                return True  # empty system
            systems.append(tuple(cur))
            cur = []
        else:
            cur.append(tok)
    if cur:
        systems.append(tuple(cur))
    elif tokens[-1] == uni_omr.SYS_TOKEN:
        return True  # trailing empty system

    if len(systems) > int(max_systems):
        return True

    # Consecutive identical systems (>= 3 in a row)
    run = 1
    for i in range(1, len(systems)):
        if len(systems[i]) >= 4 and systems[i] == systems[i - 1]:
            run += 1
            if run >= 3:
                return True
        else:
            run = 1

    # Non-consecutive heavy duplication (>= 4 identical copies on one page)
    if len(systems) >= 5:
        counts: Dict[Tuple[str, ...], int] = {}
        for s in systems:
            if len(s) >= 4:
                counts[s] = counts.get(s, 0) + 1
                if counts[s] >= 4:
                    return True
    return False


def compute_symbolic_score(
    pred_bekern: Union[str, Sequence[str]],
    *,
    tol: float = rlvr_grpo.METER_TOL,
) -> Dict[str, Any]:
    """Compute exact symbolic process score ``R_sym`` in ``[0, 1]``."""
    tokens = rlvr_grpo._as_tokens(pred_bekern)
    if not tokens:
        return {
            "r_sym": 0.0,
            "parse_ok": False,
            "n_notes": 0,
            "n_sys": 0,
            "n_valid_bars": 0,
            "n_checked_bars": 0,
            "r_meter": 0.0,
            "ties_ok": False,
            "tuplets_ok": False,
            "has_tok_loop": False,
            "has_sys_loop": False,
        }

    parse_ok, n_notes, n_valid_bars, n_checked_bars = rlvr_grpo._walk_parse_and_meter(
        tokens, tol=tol
    )
    ties_ok, tuplets_ok = rlvr_grpo._check_balanced_ties_and_tuplets(tokens)
    has_tok_loop = rlvr_grpo.has_repetition_loop(tokens)
    has_sys_loop = detect_sys_loop(tokens)
    n_sys = sum(1 for t in tokens if t == uni_omr.SYS_TOKEN) + 1

    if not parse_ok or n_notes <= 0:
        return {
            "r_sym": 0.0,
            "parse_ok": bool(parse_ok),
            "n_notes": int(n_notes),
            "n_sys": int(n_sys),
            "n_valid_bars": int(n_valid_bars),
            "n_checked_bars": int(n_checked_bars),
            "r_meter": 0.0,
            "ties_ok": bool(ties_ok),
            "tuplets_ok": bool(tuplets_ok),
            "has_tok_loop": bool(has_tok_loop),
            "has_sys_loop": bool(has_sys_loop),
        }

    r_parse = 1.0
    if n_checked_bars > 0:
        r_meter = float(n_valid_bars) / float(n_checked_bars)
    else:
        # Short excerpt / single pickup bar: neutral meter credit rather than zero
        r_meter = 0.5
    r_balance = 0.5 * float(ties_ok) + 0.5 * float(tuplets_ok)
    r_density = min(1.0, float(n_notes) / 4.0)
    loop_mult = SYM_LOOP_PENALTY if (has_tok_loop or has_sys_loop) else 1.0

    r_sym = loop_mult * r_density * (
        W_SYM_PARSE * r_parse
        + W_SYM_METER * r_meter
        + W_SYM_BALANCE * r_balance
    )
    r_sym = float(max(0.0, min(1.0, r_sym)))

    return {
        "r_sym": r_sym,
        "parse_ok": True,
        "n_notes": int(n_notes),
        "n_sys": int(n_sys),
        "n_valid_bars": int(n_valid_bars),
        "n_checked_bars": int(n_checked_bars),
        "r_meter": float(r_meter),
        "ties_ok": bool(ties_ok),
        "tuplets_ok": bool(tuplets_ok),
        "has_tok_loop": bool(has_tok_loop),
        "has_sys_loop": bool(has_sys_loop),
    }


# ---------------------------------------------------------------------------
# 2. Bekern -> MusicXML (Rust assembly-serialize + Pure-Python Fallback)
# ---------------------------------------------------------------------------

def find_assembly_serialize_bin() -> Optional[Path]:
    """Return path to ``assembly-serialize`` (or ``mscz``) binary if built, else ``None``."""
    candidates = [
        REPO_ROOT / "converter" / "assembly-serialize" / "target" / "release" / "assembly-serialize.exe",
        REPO_ROOT / "converter" / "assembly-serialize" / "target" / "release" / "assembly-serialize",
        REPO_ROOT / "converter" / "assembly-serialize" / "target" / "debug" / "assembly-serialize.exe",
        REPO_ROOT / "converter" / "assembly-serialize" / "target" / "debug" / "assembly-serialize",
        REPO_ROOT.parent / "mscz-rs" / "target" / "release" / "mscz.exe",
        REPO_ROOT.parent / "mscz-rs" / "target" / "release" / "mscz",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _count_chunk_measures(chunk: Sequence[str]) -> int:
    n_bars = sum(1 for t in chunk if t in grammar.BAR_LIKES)
    if n_bars > 0:
        last_bar = max(i for i, t in enumerate(chunk) if t in grammar.BAR_LIKES)
        has_tail = any(
            t not in grammar.BAR_LIKES and not t.startswith(("clef-", "key", "time-"))
            for t in chunk[last_bar + 1 :]
        )
        return n_bars + (1 if has_tail else 0)
    return 1 if chunk else 0


def _prepare_bekern_and_breaks(
    bekern: Union[str, Sequence[str]],
) -> Tuple[List[str], List[int]]:
    raw_toks = rlvr_grpo._as_tokens(bekern)
    if not raw_toks:
        return [], []

    chunks: List[List[str]] = []
    cur: List[str] = []
    for tok in raw_toks:
        if tok == uni_omr.SYS_TOKEN:
            if cur:
                chunks.append(cur)
                cur = []
        else:
            cur.append(tok)
    if cur:
        chunks.append(cur)
    if not chunks:
        return [], []

    new_sys_measures: List[int] = []
    if len(chunks) > 1:
        cum_m = 0
        for idx, ch in enumerate(chunks):
            if idx > 0:
                new_sys_measures.append(cum_m + 1)
            cum_m += _count_chunk_measures(ch)

    stitched, _ = eval_s31.stitch_uni_tokens(raw_toks)
    return stitched, new_sys_measures


def _inject_system_breaks(musicxml: str, new_sys_measures: Sequence[int]) -> str:
    if not new_sys_measures:
        return musicxml
    targets = {int(m) for m in new_sys_measures if int(m) > 1}
    if not targets:
        return musicxml

    def _repl(match: re.Match[str]) -> str:
        full = match.group(0)
        try:
            m_num = int(match.group(1))
        except ValueError:
            return full
        if m_num in targets:
            return full + '<print new-system="yes"/>'
        return full

    return re.sub(r'<measure\b[^>]*\bnumber="(\d+)"[^>]*>', _repl, musicxml)


def bekern_to_musicxml_fallback(bekern: Union[str, Sequence[str]]) -> Optional[str]:
    """Lightweight pure-Python ``bekern`` -> MusicXML 3.1 converter for Verovio rendering.

    Used automatically when the Rust ``assembly-serialize`` binary is unavailable
    (or as a resilient fallback). Preserves clefs, key/time signatures, measures,
    ``<sys>`` breaks, chords, rests, bar-rests, dots, accidentals, and voices.
    """
    raw_toks = rlvr_grpo._as_tokens(bekern)
    if not raw_toks:
        return None

    # Split into systems, then measures per system
    systems: List[List[str]] = []
    cur_sys: List[str] = []
    for t in raw_toks:
        if t == uni_omr.SYS_TOKEN:
            if cur_sys:
                systems.append(cur_sys)
                cur_sys = []
        else:
            cur_sys.append(t)
    if cur_sys:
        systems.append(cur_sys)
    if not systems:
        return None

    measures_xml: List[str] = []
    measure_num = 0
    cur_clef = ("G", 2)
    cur_fifths = 0
    cur_beats, cur_btype = 4, 4
    first_measure_emitted = False

    for sys_idx, sys_toks in enumerate(systems):
        # Split system into measures on BAR_LIKES
        m_chunks: List[List[str]] = []
        cur_m: List[str] = []
        for t in sys_toks:
            if t in grammar.BAR_LIKES:
                if cur_m:
                    m_chunks.append(cur_m)
                    cur_m = []
            else:
                cur_m.append(t)
        if cur_m:
            m_chunks.append(cur_m)

        for local_m_idx, m_toks in enumerate(m_chunks):
            measure_num += 1
            attr_parts: List[str] = []
            notes_parts: List[str] = []
            new_clef = None
            new_fifths = None
            new_time = None

            i = 0
            n = len(m_toks)
            cur_voice = "1"
            cur_pos = 0
            pending_chord = False
            pending_grace = False

            while i < n:
                tok = m_toks[i]
                if tok.startswith("clef-"):
                    m_c = re.match(r"^clef-([GFC])-?(\d)$", tok)
                    if m_c:
                        new_clef = (m_c.group(1), int(m_c.group(2)))
                        cur_clef = new_clef
                    i += 1
                    continue
                if tok.startswith("key"):
                    m_k = re.match(r"^key([+-]?\d+)$", tok)
                    if m_k:
                        new_fifths = int(m_k.group(1))
                        cur_fifths = new_fifths
                    i += 1
                    continue
                if tok.startswith("time-"):
                    m_t = re.match(r"^time-(\d+)[/-](\d+)$", tok)
                    if m_t:
                        new_time = (int(m_t.group(1)), int(m_t.group(2)))
                        cur_beats, cur_btype = new_time
                    elif tok in ("time-C", "time-common"):
                        new_time = (4, 4)
                        cur_beats, cur_btype = new_time
                    elif tok in ("time-cut", "time-alla-breve"):
                        new_time = (2, 2)
                        cur_beats, cur_btype = new_time
                    i += 1
                    continue
                if tok.startswith("voice-"):
                    if cur_pos > 0:
                        notes_parts.append(f"<backup><duration>{cur_pos}</duration></backup>")
                        cur_pos = 0
                    cur_voice = tok.split("-", 1)[1] or "1"
                    i += 1
                    continue
                if tok == grammar.CHORD:
                    pending_chord = True
                    i += 1
                    continue
                if tok == grammar.GRACE:
                    pending_grace = True
                    i += 1
                    continue
                if tok == grammar.BAR_REST:
                    dur_val = int(round(24 * cur_beats * (4.0 / max(1, cur_btype))))
                    notes_parts.append(
                        f"<note><rest measure=\"yes\"/><duration>{dur_val}</duration>"
                        f"<voice>{xml_escape(cur_voice)}</voice></note>"
                    )
                    cur_pos += dur_val
                    pending_chord = False
                    pending_grace = False
                    i += 1
                    continue
                if tok == grammar.REST:
                    i += 1
                    dur_tok = "quarter"
                    if i < n and m_toks[i] in _DUR_DIVISIONS:
                        dur_tok = m_toks[i]
                        i += 1
                    n_dots = 0
                    while i < n and m_toks[i] == grammar.DOT:
                        n_dots += 1
                        i += 1
                    base_d, xml_type = _DUR_DIVISIONS[dur_tok]
                    dur_val = base_d
                    add = base_d
                    for _ in range(n_dots):
                        add = max(1, add // 2)
                        dur_val += add
                    dot_xml = "<dot/>" * n_dots
                    notes_parts.append(
                        f"<note><rest/><duration>{dur_val}</duration>"
                        f"<voice>{xml_escape(cur_voice)}</voice>"
                        f"<type>{xml_type}</type>{dot_xml}</note>"
                    )
                    cur_pos += dur_val
                    pending_chord = False
                    pending_grace = False
                    continue
                if tok in _DUR_DIVISIONS and i + 2 < n and m_toks[i + 1] in grammar.STEPS and m_toks[i + 2] in grammar.OCTAVES:
                    base_d, xml_type = _DUR_DIVISIONS[tok]
                    step = m_toks[i + 1]
                    octave = m_toks[i + 2]
                    alter_val: Optional[int] = None
                    n_dots = 0
                    j = i + 3
                    while j < n:
                        tt = m_toks[j]
                        if tt in grammar.ALTERS:
                            try:
                                alter_val = int(tt[1:])
                            except ValueError:
                                pass
                            j += 1
                        elif tt == grammar.DOT:
                            n_dots += 1
                            j += 1
                        elif tt in grammar.MODS or tt in (grammar.TIE_START, grammar.TIE_STOP):
                            j += 1
                        else:
                            break
                    dur_val = base_d
                    add = base_d
                    for _ in range(n_dots):
                        add = max(1, add // 2)
                        dur_val += add
                    chord_xml = "<chord/>" if pending_chord else ""
                    grace_xml = "<grace/>" if pending_grace else ""
                    alter_xml = f"<alter>{alter_val}</alter>" if alter_val is not None else ""
                    dot_xml = "<dot/>" * n_dots
                    dur_tag = "" if pending_grace else f"<duration>{dur_val}</duration>"
                    notes_parts.append(
                        f"<note>{grace_xml}{chord_xml}"
                        f"<pitch><step>{step}</step>{alter_xml}<octave>{octave}</octave></pitch>"
                        f"{dur_tag}<voice>{xml_escape(cur_voice)}</voice>"
                        f"<type>{xml_type}</type>{dot_xml}</note>"
                    )
                    if not pending_chord and not pending_grace:
                        cur_pos += dur_val
                    pending_chord = False
                    pending_grace = False
                    i = j
                    continue
                i += 1

            if not first_measure_emitted:
                attr_parts.append("<divisions>24</divisions>")
                attr_parts.append(f"<key><fifths>{cur_fifths}</fifths></key>")
                attr_parts.append(f"<time><beats>{cur_beats}</beats><beat-type>{cur_btype}</beat-type></time>")
                attr_parts.append(f"<clef><sign>{cur_clef[0]}</sign><line>{cur_clef[1]}</line></clef>")
                first_measure_emitted = True
            else:
                if new_fifths is not None:
                    attr_parts.append(f"<key><fifths>{new_fifths}</fifths></key>")
                if new_time is not None:
                    attr_parts.append(f"<time><beats>{new_time[0]}</beats><beat-type>{new_time[1]}</beat-type></time>")
                if new_clef is not None:
                    attr_parts.append(f"<clef><sign>{new_clef[0]}</sign><line>{new_clef[1]}</line></clef>")

            print_xml = '<print new-system="yes"/>' if (sys_idx > 0 and local_m_idx == 0) else ""
            attr_xml = f"<attributes>{''.join(attr_parts)}</attributes>" if attr_parts else ""
            if not notes_parts:
                dur_val = int(round(24 * cur_beats * (4.0 / max(1, cur_btype))))
                notes_parts.append(f"<note><rest measure=\"yes\"/><duration>{dur_val}</duration><voice>1</voice></note>")
            measures_xml.append(f'<measure number="{measure_num}">{print_xml}{attr_xml}{"".join(notes_parts)}</measure>')

    if not measures_xml:
        return None

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<score-partwise version="3.1">\n'
        '  <part-list><score-part id="P1"><part-name>Music</part-name></score-part></part-list>\n'
        '  <part id="P1">\n'
        + "\n".join(measures_xml)
        + "\n  </part>\n</score-partwise>\n"
    )


def bekern_to_musicxml(
    bekern: Union[str, Sequence[str]],
    *,
    aser_bin: Optional[Path] = None,
    use_fallback: bool = True,
    timeout_s: float = 15.0,
) -> Optional[str]:
    """Convert ``bekern`` -> MusicXML via Rust ``assembly-serialize`` with Python fallback."""
    stitched, new_sys_measures = _prepare_bekern_and_breaks(bekern)
    if not stitched:
        return None

    exe = aser_bin if aser_bin is not None else find_assembly_serialize_bin()
    if exe is not None and exe.exists():
        with tempfile.TemporaryDirectory(prefix="dual_ver_") as td:
            td_path = Path(td)
            in_txt = td_path / "pred.txt"
            out_dir = td_path / "out"
            in_txt.write_text(" ".join(stitched) + "\n", encoding="utf-8")
            try:
                proc = subprocess.run(
                    [
                        str(exe),
                        "bekern2musicxml",
                        "--input",
                        str(in_txt),
                        "--out",
                        str(out_dir),
                        "--stem",
                        "pred",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                )
                xml_path = out_dir / "pred.musicxml"
                if proc.returncode == 0 and xml_path.exists():
                    xml_text = xml_path.read_text(encoding="utf-8", errors="replace")
                    if xml_text.strip():
                        if new_sys_measures:
                            xml_text = _inject_system_breaks(xml_text, new_sys_measures)
                        return xml_text
            except (subprocess.TimeoutExpired, OSError):
                pass

    if use_fallback:
        return bekern_to_musicxml_fallback(bekern)
    return None


_VEROVIO_LOCK = threading.Lock()
_VEROVIO_TK: Any = None


def _get_verovio_toolkit() -> Any:
    """Return a single thread-safe Verovio toolkit with explicit resource path.

    Instantiating ``verovio.toolkit()`` inside a ``ThreadPoolExecutor`` worker
    thread fails to locate ``verovio/data`` fonts (Bravura/Leipzig), causing
    100% silent render failures. Reusing one instance under ``_VEROVIO_LOCK``
    with an explicit ``setResourcePath`` fixes both the thread-init bug and
    avoids repeated font-table reloads.
    """
    global _VEROVIO_TK
    if _VEROVIO_TK is None:
        try:
            verovio.enableLog(verovio.LOG_OFF)
        except Exception:
            pass
        data_dir = Path(verovio.__file__).resolve().parent / "data"
        if data_dir.is_dir():
            tk = verovio.toolkit(False)
            tk.setResourcePath(str(data_dir))
        else:
            tk = verovio.toolkit()
        _VEROVIO_TK = tk
    return _VEROVIO_TK


def render_musicxml_to_gray(
    musicxml: str,
    *,
    page_width: int = 2100,
    scale: int = 40,
    use_encoded_breaks: bool = False,
) -> Optional[np.ndarray]:
    """Render MusicXML -> SVG (``verovio``) -> grayscale float32 ``[0, 1]`` (``0=ink, 1=paper``)."""
    if verovio is None or not musicxml or not musicxml.strip():
        return None
    if resvg_py is None and cairosvg is None:
        return None

    opts: Dict[str, Any] = {
        "pageWidth": int(page_width),
        "pageHeight": 60000,
        "scale": int(scale),
        "adjustPageHeight": True,
        "header": "none",
        "footer": "none",
        "breaks": "encoded" if use_encoded_breaks else "auto",
    }
    try:
        svgs: List[str] = []
        with _VEROVIO_LOCK:
            tk = _get_verovio_toolkit()
            tk.setOptions(opts)
            if not tk.loadData(musicxml):
                return None
            n_pages = int(tk.getPageCount())
            if n_pages < 1:
                return None
            for p_idx in range(1, n_pages + 1):
                svg = tk.renderToSVG(p_idx)
                if svg:
                    svgs.append(svg)

        pages: List[np.ndarray] = []
        for svg in svgs:
            if resvg_py is not None:
                png_bytes = resvg_py.svg_to_bytes(svg_string=svg, background="#ffffff")
            else:
                png_bytes = cairosvg.svg2png(bytestring=svg.encode("utf-8"), background_color="#ffffff")
            im = Image.open(io.BytesIO(png_bytes)).convert("L")
            pages.append(np.asarray(im, dtype=np.float32) / 255.0)

        if not pages:
            return None
        if len(pages) == 1:
            return pages[0]
        max_w = max(p.shape[1] for p in pages)
        padded: List[np.ndarray] = []
        for p in pages:
            if p.shape[1] < max_w:
                pad = np.ones((p.shape[0], max_w - p.shape[1]), dtype=np.float32)
                p = np.concatenate([p, pad], axis=1)
            padded.append(p)
        return np.concatenate(padded, axis=0)
    except Exception:
        return None


def render_bekern_to_gray(
    bekern: Union[str, Sequence[str]],
    *,
    aser_bin: Optional[Path] = None,
    use_fallback: bool = True,
    page_width: int = 2100,
    scale: int = 40,
) -> Optional[np.ndarray]:
    """Render ``bekern`` -> MusicXML -> Verovio SVG -> float32 grayscale ``[0, 1]``."""
    raw_toks = rlvr_grpo._as_tokens(bekern)
    if not raw_toks:
        return None
    has_sys = uni_omr.SYS_TOKEN in raw_toks
    xml_text = bekern_to_musicxml(raw_toks, aser_bin=aser_bin, use_fallback=use_fallback)
    if xml_text is None:
        return None
    use_encoded = has_sys and ('new-system="yes"' in xml_text)
    return render_musicxml_to_gray(
        xml_text,
        page_width=page_width,
        scale=scale,
        use_encoded_breaks=use_encoded,
    )


# ---------------------------------------------------------------------------
# 3. Style- & Spacing-Invariant Visual-Cycle Verifier (R_vis)
# ---------------------------------------------------------------------------

def to_gray_float01(img: Any) -> np.ndarray:
    """Convert Path, PIL Image, numpy array, or torch Tensor to ``(H, W)`` float32 in ``[0, 1]``."""
    if isinstance(img, (str, Path)):
        return np.asarray(Image.open(img).convert("L"), dtype=np.float32) / 255.0
    if isinstance(img, Image.Image):
        return np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    if hasattr(img, "detach") and hasattr(img, "cpu"):
        t = img.detach().cpu().float()
        if t.ndim == 4:
            t = t[0]
        if t.ndim == 3:
            if t.shape[0] in (1, 3, 4):
                t = t[:3].mean(dim=0)
            else:
                t = t[..., :3].mean(dim=-1)
        arr = t.numpy()
        if arr.min() < -0.05:
            arr = arr * 0.25 + 0.85
        elif arr.max() > 1.5:
            arr = arr / 255.0
        return np.clip(arr.astype(np.float32), 0.0, 1.0)

    arr = np.asarray(img)
    if arr.ndim == 3:
        if arr.shape[2] in (1, 3, 4):
            arr = arr[..., :3].mean(axis=2)
        elif arr.shape[0] in (1, 3, 4):
            arr = arr[:3].mean(axis=0)
    arr = arr.astype(np.float32)
    if arr.size == 0:
        return np.ones((1, 1), dtype=np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0
    elif arr.min() < -0.05:
        arr = arr * 0.25 + 0.85
    return np.clip(arr, 0.0, 1.0)


def extract_style_invariant_ink(
    img: Any,
    *,
    max_side: int = 512,
    ink_thresh: float = 0.12,
) -> Optional[np.ndarray]:
    """Extract illumination-, paper-color-, and contrast-invariant cropped ink map in ``[0, 1]``.

    1. Downscales large images to ``max_side`` for fast, noise-robust filtering.
    2. Estimates local paper background via large-kernel smoothing (removes
       paper yellowing, sepia tint, shadows, and uneven phone lighting).
    3. Computes relative ink darkness ``(bg - gray) / max(bg, 0.25)`` and
       rescales foreground ink by its 98th percentile (invariant to iron-gall
       brown vs crisp black engraved ink).
    4. Tightly crops to the robust foreground ink bounding box (ignoring
       1% extreme border specks / desk edges).
    """
    gray = to_gray_float01(img)
    h, w = gray.shape
    if h < 4 or w < 4:
        return None

    scale = min(1.0, float(max_side) / float(max(h, w)))
    if scale < 1.0:
        nh = max(4, int(round(h * scale)))
        nw = max(4, int(round(w * scale)))
        pil_g = Image.fromarray(np.clip(gray * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
        gray = np.asarray(pil_g.resize((nw, nh), Image.BILINEAR), dtype=np.float32) / 255.0
        h, w = gray.shape

    # Local paper illumination estimate via coarse 90th-percentile envelope
    # (immune to ink density AND removes uneven lighting / paper yellowing).
    cell = max(16, min(h, w) // 10)
    gh = max(2, (h + cell - 1) // cell)
    gw = max(2, (w + cell - 1) // cell)
    bg_grid = np.empty((gh, gw), dtype=np.float32)
    for r in range(gh):
        r0, r1 = r * cell, min(h, (r + 1) * cell)
        for c in range(gw):
            c0, c1 = c * cell, min(w, (c + 1) * cell)
            bg_grid[r, c] = float(np.percentile(gray[r0:r1, c0:c1], 90))
    pil_bg = Image.fromarray(np.clip(bg_grid * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
    bg = np.asarray(pil_bg.resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0

    # Pure local background subtraction removes uneven illumination & paper yellowing
    ink = np.clip((bg - gray) / np.maximum(bg, 0.20), 0.0, 1.0)
    # Suppress tiny sub-5% background interpolation residuals to exact 0.0
    ink = np.where(ink > 0.05, ink, 0.0).astype(np.float32)

    fg_vals = ink[ink > 0.06]
    if fg_vals.size < 16:
        return None
    p98 = float(np.percentile(fg_vals, 98))
    if p98 < 1e-4:
        return None
    ink = np.clip(ink / p98, 0.0, 1.0)

    # Robust bounding box: ignore isolated 0.5% border specks/desk artifacts
    mask = ink > float(ink_thresh)
    ys, xs = np.where(mask)
    if ys.size < 16 or xs.size < 16:
        return None
    y0 = int(np.percentile(ys, 0.5))
    y1 = int(np.percentile(ys, 99.5)) + 1
    x0 = int(np.percentile(xs, 0.5))
    x1 = int(np.percentile(xs, 99.5)) + 1
    y0, y1 = max(0, y0), min(h, max(y0 + 2, y1))
    x0, x1 = max(0, x0), min(w, max(x0 + 2, x1))
    return ink[y0:y1, x0:x1]


def extract_visual_profiles(
    ink: np.ndarray,
    *,
    n_cols: int = DEFAULT_ONSET_COLS,
    n_bands: int = DEFAULT_PITCH_BANDS,
    n_vert: int = DEFAULT_VERT_BINS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract staff-normalized horizontal onset columns and vertical pitch-density profile.

    Returns:
      - ``onset_cols``: ``(n_cols, n_bands)`` float32 — staff-line-suppressed
        vertical ink distribution at each horizontal onset column.
      - ``vert_prof``: ``(n_vert,)`` float32 — vertical ink density profile
        capturing system count, spacing, and pitch-density envelope.
    """
    # 1. Vertical pitch-density profile (mean ink across width, resampled to n_vert)
    row_mean = ink.mean(axis=1).astype(np.float32)
    x_old = np.linspace(0.0, 1.0, len(row_mean), dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, int(n_vert), dtype=np.float32)
    vert_prof = np.interp(x_new, x_old, row_mean).astype(np.float32)

    # 2. Staff-line suppression for horizontal onset/barline projection:
    #    Staff lines have high horizontal continuity (high row 30th percentile),
    #    whereas noteheads, stems, beams, and barlines are local spikes above it.
    row_base = np.percentile(ink, 30, axis=1, keepdims=True).astype(np.float32)
    onset_ink = np.clip(ink - row_base, 0.0, 1.0)
    if float(onset_ink.max()) < 1e-4:
        onset_ink = ink

    # Resize onset_ink to (n_bands, n_cols) via area/bilinear pooling
    pil_on = Image.fromarray(np.clip(onset_ink * 255.0, 0.0, 255.0).astype(np.uint8), mode="L")
    pooled = np.asarray(
        pil_on.resize((int(n_cols), int(n_bands)), Image.BILINEAR), dtype=np.float32
    ) / 255.0
    onset_cols = pooled.T.copy()  # (n_cols, n_bands)
    return onset_cols, vert_prof


def _emd_1d_vectors(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized 1D Earth Mover's Distance (Wasserstein-1) in ``[0, 1]``."""
    sa = float(a.sum())
    sb = float(b.sum())
    if sa < 1e-8 and sb < 1e-8:
        return 0.0
    if sa < 1e-8 or sb < 1e-8:
        return 1.0
    ca = np.cumsum(a / sa)
    cb = np.cumsum(b / sb)
    return float(np.mean(np.abs(ca - cb)))


def dtw_onset_emd_distance(
    cols_a: np.ndarray,
    cols_b: np.ndarray,
    *,
    band: int = DEFAULT_DTW_BAND,
    mass_weight: float = 0.35,
) -> float:
    """Sakoe-Chiba banded DTW distance between onset-column sequences using 1D EMD + mass cost.

    Args:
        cols_a: ``(N, B)`` float32 non-negative onset-column pitch-band arrays.
        cols_b: ``(M, B)`` float32 non-negative onset-column pitch-band arrays.
        band: Sakoe-Chiba warping window half-width.
        mass_weight: weight on relative column ink-mass mismatch so empty
            columns cannot freely match dense note columns.

    Returns:
        Mean per-step DTW alignment cost ``>= 0.0`` (``0.0`` for identical sequences).
    """
    n, b_dim = cols_a.shape
    m = cols_b.shape[0]
    if n == 0 or m == 0:
        return 1.0

    # Precompute column masses (normalized by mean column mass) and CDFs
    mass_a = cols_a.sum(axis=1)
    mass_b = cols_b.sum(axis=1)
    mean_ma = float(mass_a.mean()) + 1e-8
    mean_mb = float(mass_b.mean()) + 1e-8
    rel_ma = mass_a / mean_ma
    rel_mb = mass_b / mean_mb

    # Smooth vertical distributions slightly so near-empty columns vary continuously
    eps_a = 0.02 * mean_ma / max(1, b_dim)
    eps_b = 0.02 * mean_mb / max(1, b_dim)
    pa = cols_a + eps_a
    pb = cols_b + eps_b
    cdf_a = np.cumsum(pa / np.maximum(pa.sum(axis=1, keepdims=True), 1e-8), axis=1)
    cdf_b = np.cumsum(pb / np.maximum(pb.sum(axis=1, keepdims=True), 1e-8), axis=1)

    # Pairwise 1D EMD matrix (N, M): mean |cdf_a[i] - cdf_b[j]|
    emd_mat = np.abs(cdf_a[:, None, :] - cdf_b[None, :, :]).mean(axis=2)

    # Relative ink-mass mismatch in [0, 1] (penalizes empty-vs-dense matching smoothly)
    mass_diff = np.abs(rel_ma[:, None] - rel_mb[None, :]) / (
        rel_ma[:, None] + rel_mb[None, :] + 0.15
    )
    cost = (1.0 - mass_weight) * emd_mat + mass_weight * mass_diff

    w = max(int(band), abs(n - m))
    inf = 1e9
    dp = np.full((n + 1, m + 1), inf, dtype=np.float64)
    dp[0, 0] = 0.0

    for i in range(1, n + 1):
        j_lo = max(1, i - w)
        j_hi = min(m, i + w)
        row_c = cost[i - 1]
        for j in range(j_lo, j_hi + 1):
            dp[i, j] = float(row_c[j - 1]) + min(
                dp[i - 1, j],
                dp[i, j - 1],
                dp[i - 1, j - 1],
            )

    path_len = float(max(n, m))
    return float(dp[n, m] / path_len)


def _pearson_clamped01(a: np.ndarray, b: np.ndarray) -> float:
    ac = a.astype(np.float64) - float(a.mean())
    bc = b.astype(np.float64) - float(b.mean())
    na = float(np.linalg.norm(ac))
    nb = float(np.linalg.norm(bc))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(max(0.0, min(1.0, float(np.dot(ac, bc) / (na * nb)))))


def compute_visual_cycle_score(
    real_img: Any,
    rendered_img: Any,
    *,
    n_cols: int = DEFAULT_ONSET_COLS,
    n_bands: int = DEFAULT_PITCH_BANDS,
    n_vert: int = DEFAULT_VERT_BINS,
    dtw_band: int = DEFAULT_DTW_BAND,
    dtw_alpha: float = DEFAULT_DTW_ALPHA,
    vert_emd_alpha: float = DEFAULT_VERT_EMD_ALPHA,
) -> Dict[str, Any]:
    """Compute style- & spacing-invariant visual similarity ``R_vis`` in ``[0, 1]``."""
    ink_real = extract_style_invariant_ink(real_img)
    ink_rend = extract_style_invariant_ink(rendered_img)
    if ink_real is None or ink_rend is None:
        return {
            "r_vis": 0.0,
            "s_dtw_emd": 0.0,
            "dtw_emd_dist": 1.0,
            "v_corr": 0.0,
            "v_emd_sim": 0.0,
            "v_emd_dist": 1.0,
            "visual_ok": False,
        }

    cols_real, vert_real = extract_visual_profiles(
        ink_real, n_cols=n_cols, n_bands=n_bands, n_vert=n_vert
    )
    cols_rend, vert_rend = extract_visual_profiles(
        ink_rend, n_cols=n_cols, n_bands=n_bands, n_vert=n_vert
    )

    dtw_dist = dtw_onset_emd_distance(cols_real, cols_rend, band=dtw_band)
    s_dtw_emd = float(math.exp(-float(dtw_alpha) * dtw_dist))
    v_corr = _pearson_clamped01(vert_real, vert_rend)
    v_emd_dist = _emd_1d_vectors(vert_real, vert_rend)
    v_emd_sim = float(math.exp(-float(vert_emd_alpha) * v_emd_dist))

    r_vis = (
        W_VIS_DTW_EMD * s_dtw_emd
        + W_VIS_VERT_CORR * v_corr
        + W_VIS_VERT_EMD * v_emd_sim
    )
    r_vis = float(max(0.0, min(1.0, r_vis)))

    return {
        "r_vis": r_vis,
        "s_dtw_emd": float(s_dtw_emd),
        "dtw_emd_dist": float(dtw_dist),
        "v_corr": float(v_corr),
        "v_emd_sim": float(v_emd_sim),
        "v_emd_dist": float(v_emd_dist),
        "visual_ok": True,
    }


# ---------------------------------------------------------------------------
# 4. DualVerifier Class
# ---------------------------------------------------------------------------

@dataclass
class DualVerifierConfig:
    """Configuration for ``DualVerifier``."""

    aser_bin: Optional[Path] = None
    use_python_musicxml_fallback: bool = True
    page_width: int = 2100
    render_scale: int = 40
    n_cols: int = DEFAULT_ONSET_COLS
    n_bands: int = DEFAULT_PITCH_BANDS
    n_vert: int = DEFAULT_VERT_BINS
    dtw_band: int = DEFAULT_DTW_BAND
    dtw_alpha: float = DEFAULT_DTW_ALPHA
    vert_emd_alpha: float = DEFAULT_VERT_EMD_ALPHA
    sym_floor: float = DUAL_SYM_FLOOR
    vis_weight: float = DUAL_VIS_WEIGHT


class DualVerifier:
    """Dual-Verified Render-Cycle Verifier combining ``R_sym`` and ``R_vis``.

    Combined score:
        ``R_dual = R_sym * (0.2 + 0.8 * R_vis)``
    """

    def __init__(
        self,
        cfg: Optional[DualVerifierConfig] = None,
        *,
        render_fn: Optional[Callable[[Sequence[str]], Optional[np.ndarray]]] = None,
    ) -> None:
        self.cfg = cfg if cfg is not None else DualVerifierConfig()
        self._render_fn = render_fn
        self._real_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def _extract_real_profiles(self, real_img: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        ink_real = extract_style_invariant_ink(real_img)
        if ink_real is None:
            return None
        return extract_visual_profiles(
            ink_real,
            n_cols=self.cfg.n_cols,
            n_bands=self.cfg.n_bands,
            n_vert=self.cfg.n_vert,
        )

    def render(self, pred_bekern: Union[str, Sequence[str]]) -> Optional[np.ndarray]:
        toks = rlvr_grpo._as_tokens(pred_bekern)
        if not toks:
            return None
        if self._render_fn is not None:
            return self._render_fn(toks)
        return render_bekern_to_gray(
            toks,
            aser_bin=self.cfg.aser_bin,
            use_fallback=self.cfg.use_python_musicxml_fallback,
            page_width=self.cfg.page_width,
            scale=self.cfg.render_scale,
        )

    def score_visual_from_profiles(
        self,
        real_profiles: Optional[Tuple[np.ndarray, np.ndarray]],
        rendered_img: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        if real_profiles is None or rendered_img is None:
            return {
                "r_vis": 0.0,
                "s_dtw_emd": 0.0,
                "dtw_emd_dist": 1.0,
                "v_corr": 0.0,
                "v_emd_sim": 0.0,
                "v_emd_dist": 1.0,
                "visual_ok": False,
            }
        ink_rend = extract_style_invariant_ink(rendered_img)
        if ink_rend is None:
            return {
                "r_vis": 0.0,
                "s_dtw_emd": 0.0,
                "dtw_emd_dist": 1.0,
                "v_corr": 0.0,
                "v_emd_sim": 0.0,
                "v_emd_dist": 1.0,
                "visual_ok": False,
            }
        cols_real, vert_real = real_profiles
        cols_rend, vert_rend = extract_visual_profiles(
            ink_rend,
            n_cols=self.cfg.n_cols,
            n_bands=self.cfg.n_bands,
            n_vert=self.cfg.n_vert,
        )
        dtw_dist = dtw_onset_emd_distance(cols_real, cols_rend, band=self.cfg.dtw_band)
        s_dtw_emd = float(math.exp(-float(self.cfg.dtw_alpha) * dtw_dist))
        v_corr = _pearson_clamped01(vert_real, vert_rend)
        v_emd_dist = _emd_1d_vectors(vert_real, vert_rend)
        v_emd_sim = float(math.exp(-float(self.cfg.vert_emd_alpha) * v_emd_dist))
        r_vis = float(
            max(
                0.0,
                min(
                    1.0,
                    W_VIS_DTW_EMD * s_dtw_emd
                    + W_VIS_VERT_CORR * v_corr
                    + W_VIS_VERT_EMD * v_emd_sim,
                ),
            )
        )
        return {
            "r_vis": r_vis,
            "s_dtw_emd": float(s_dtw_emd),
            "dtw_emd_dist": float(dtw_dist),
            "v_corr": float(v_corr),
            "v_emd_sim": float(v_emd_sim),
            "v_emd_dist": float(v_emd_dist),
            "visual_ok": True,
        }

    def score(
        self,
        real_img: Any,
        pred_bekern: Union[str, Sequence[str]],
        *,
        precomputed_real_profiles: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """Compute ``R_sym``, ``R_vis``, and ``R_dual = R_sym * (0.2 + 0.8 * R_vis)``."""
        sym = compute_symbolic_score(pred_bekern)
        r_sym = float(sym["r_sym"])

        # Skip expensive rendering when symbolic parse completely failed
        if r_sym <= 0.0:
            return {
                "reward": 0.0,
                "r_dual": 0.0,
                "r_sym": 0.0,
                "r_vis": 0.0,
                "render_ok": False,
                **sym,
                "s_dtw_emd": 0.0,
                "dtw_emd_dist": 1.0,
                "v_corr": 0.0,
                "v_emd_sim": 0.0,
                "v_emd_dist": 1.0,
                "visual_ok": False,
            }

        real_prof = (
            precomputed_real_profiles
            if precomputed_real_profiles is not None
            else self._extract_real_profiles(real_img)
        )
        rendered = self.render(pred_bekern)
        render_ok = rendered is not None
        vis = self.score_visual_from_profiles(real_prof, rendered)
        r_vis = float(vis["r_vis"])
        r_dual = float(
            max(0.0, min(1.0, r_sym * (self.cfg.sym_floor + self.cfg.vis_weight * r_vis)))
        )

        return {
            "reward": r_dual,
            "r_dual": r_dual,
            "r_sym": r_sym,
            "r_vis": r_vis,
            "render_ok": bool(render_ok),
            **sym,
            **vis,
        }

    def score_batch_for_page(
        self,
        real_img: Any,
        candidates: Sequence[Union[str, Sequence[str]]],
    ) -> List[Dict[str, Any]]:
        """Score multiple candidate rollouts for the same page (caches real-page profile)."""
        real_prof = self._extract_real_profiles(real_img)
        return [
            self.score(real_img, cand, precomputed_real_profiles=real_prof)
            for cand in candidates
        ]
