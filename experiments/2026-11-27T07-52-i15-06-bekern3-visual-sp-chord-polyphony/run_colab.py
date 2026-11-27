#!/usr/bin/env python3
"""Colab L4 / Modal L4 runner for Wave 13 Exp i15-06 (bekern3 Architect 1):
2026-11-27T07-52-i15-06-bekern3-visual-sp-chord-polyphony

2-arm 1,200-step BF16 ablation on `UniOMRv2Net(small_45m)` (`iter10_5_proxy_2k`):
  - arm0_bekern_v2a   : Standard `bekern+v2` Phase A sounding pitch & rhythm baseline
  - arm1_bekern3_core : `bekern3` Core Visual-Pitch, Rhythm & Polyphony:
      (1) Visual staff-position `sp-14..sp+20` (+ lossless octave-shift escape) &
          5 printed-only `acc-*` tokens + exact `Ratio<i64>` (`Fraction`) resolver
      (2) Duration-first implicit chords + fused dotted durations (`quarter.`, etc.)
      (3) Descending-`mean(sp)` polyphonic `<v>` + `skip <DUR>` & rational `d0..d9 /`
          tuplets and meters

Measures:
  (a) Deterministic MusicXML round-trip Oracle on `suite31_hw` (>= 99.5% gate)
  (b) Exact token-length reduction across `proxy_train`, `proxy_val`, `suite31_hw` (>= 30% gate)
  (c) Held-out `suite31_hw` (`HWM` & `printed`, clean + photo) Note F1 + MTP-4
      effective notes/sec speculative decoding throughput

Uploads `evals/<EXP_ID>.json` and `runs/<EXP_ID>.json` to HF `gregoreesmaa/sheet-music-converter-omr`.
Resume-safe (5-minute atomic checkpoints + per-arm `arm_done.json`). Never prints `HF_TOKEN`.
"""

from __future__ import annotations

from collections import Counter
import contextlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fractions import Fraction
import gc
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tarfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

EXP_ID = "2026-11-27T07-52-i15-06-bekern3-visual-sp-chord-polyphony"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_PROXY_TAR = "datasets/iter10_5_proxy_2k.tar"
HF_SUITE31_TAR = "datasets/suite31_hw_eval.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

HF_TOKEN_FILE = Path("/tmp/.hf_token")
STEPS_PER_ARM = int(os.environ.get("I15_06_STEPS", "1200"))
SEED = 42
GRAD_ACCUM = 4
LR = 3.0e-4
MIN_LR = 3.0e-5
WARMUP_STEPS = 80
MTP_LOSS_WEIGHT = 0.30
FULLPAGE_MAX_DIM = 1280
MAX_SEQ_LEN = 1536
EVAL_MAX_LEN = 1536
GPU_EMA_DECAY = 0.99
SAVE_EVERY_MINUTES = 5.0
VAL_LIMIT = int(os.environ.get("I15_06_VAL_LIMIT", "64"))
SUITE31_LIMIT = int(os.environ.get("I15_06_SUITE31_LIMIT", "0"))  # 0 = all 45 pieces
MTP_BENCH_PAGES = int(os.environ.get("I15_06_MTP_PAGES", "16"))

# ---------------------------------------------------------------------------
# bekern3_core Core Constants & Exact Ratio<i64> (Fraction) Music Theory
# ---------------------------------------------------------------------------

STEP_TO_IDX: Dict[str, int] = {
    "C": 0, "D": 1, "E": 2, "F": 3, "G": 4, "A": 5, "B": 6,
}
IDX_TO_STEP: Tuple[str, ...] = ("C", "D", "E", "F", "G", "A", "B")
SHARP_ORDER: Tuple[str, ...] = ("F", "C", "G", "D", "A", "E", "B")
FLAT_ORDER: Tuple[str, ...] = ("B", "E", "A", "D", "G", "C", "F")

STEPS_SET = frozenset(STEP_TO_IDX.keys())
OCTAVES_SET = frozenset(str(o) for o in range(-1, 9))
ALTER_TO_INT: Dict[str, int] = {"a-2": -2, "a-1": -1, "a1": 1, "a2": 2}
INT_TO_ALTER: Dict[int, str] = {-2: "a-2", -1: "a-1", 1: "a1", 2: "a2"}

ACC_TO_INT: Dict[str, int] = {
    "acc-2": -2,
    "acc-1": -1,
    "acc0": 0,
    "acc+1": 1,
    "acc+2": 2,
}
INT_TO_ACC: Dict[int, str] = {v: k for k, v in ACC_TO_INT.items()}

BASE_DURATIONS: Tuple[str, ...] = (
    "breve", "whole", "half", "quarter", "eighth", "16th", "32nd", "64th", "128th",
)
BASE_DURATIONS_SET = frozenset(BASE_DURATIONS)
DUR_TO_FRACTION: Dict[str, Fraction] = {
    "breve": Fraction(2, 1),
    "whole": Fraction(1, 1),
    "half": Fraction(1, 2),
    "quarter": Fraction(1, 4),
    "eighth": Fraction(1, 8),
    "16th": Fraction(1, 16),
    "32nd": Fraction(1, 32),
    "64th": Fraction(1, 64),
    "128th": Fraction(1, 128),
}

# Fused dotted durations (zero-lag): base, single dot (.), double dot (..)
FUSED_DUR_TOKENS: List[str] = []
FUSED_DUR_PARSE: Dict[str, Tuple[str, int]] = {}
for _d in BASE_DURATIONS:
    FUSED_DUR_PARSE[_d] = (_d, 0)
    for _dots, _suf in ((1, "."), (2, "..")):
        _tok = f"{_d}{_suf}"
        FUSED_DUR_TOKENS.append(_tok)
        FUSED_DUR_PARSE[_tok] = (_d, _dots)
ALL_FUSED_DURS_SET = frozenset(FUSED_DUR_PARSE.keys())

# Core visual staff-position range: sp-14..sp+20 (35 tokens) + octave-shift escapes
SP_CORE_MIN = -14
SP_CORE_MAX = 20
SP_CORE_TOKENS: List[str] = [f"sp{k:+d}" for k in range(SP_CORE_MIN, SP_CORE_MAX + 1)]
SP_OCT_DN = "sp-oct-dn"
SP_OCT_UP = "sp-oct-up"
ACC_TOKENS: List[str] = ["acc-2", "acc-1", "acc0", "acc+1", "acc+2"]

# Polyphony + Rational grammar tokens
VOICE_SEP = "<v>"
SKIP_TOK = "skip"
TIME_PREFIX = "time"
TUP_PREFIX = "tup"
TUP_END = "tup-end"
RAT_SLASH = "/"
DIGIT_TOKENS: Tuple[str, ...] = tuple(f"d{i}" for i in range(10))
DIGIT_TO_CHAR: Dict[str, str] = {f"d{i}": str(i) for i in range(10)}
CHAR_TO_DIGIT: Dict[str, str] = {str(i): f"d{i}" for i in range(10)}

BAR_RESET_TOKENS = frozenset(
    ("bar", "bar-light-heavy", "bar-repeat-right", "bar-double", "bar-repeat-left")
)
POST_NOTE_MARKS = frozenset(
    (
        "tie-start", "tie-stop", "slur-start", "slur-stop",
        "stacc", "accent", "tenuto", "fermata",
        "art-arpeggiate", "art-trill", "art-tremolo", "art-mordent",
        "art-turn", "art-staccatissimo", "art-up-bow", "art-down-bow",
        "art-soft-accent",
    )
)

SP_PATTERN = re.compile(r"^sp([+-]\d+)$")
CLEF_PATTERN = re.compile(r"^clef-([A-Za-z]+)-(\d+)(?:-(8vb|8va|15mb|15ma))?$")
KEY_PATTERN = re.compile(r"^key([+-]?\d+)$")
TIME_V2_PATTERN = re.compile(r"^time-(\d+)-(\d+)$")
TUPLET_V2_PATTERN = re.compile(r"^tuplet-start-(\d+)-(\d+)$")
VOICE_V2_PATTERN = re.compile(r"^voice-(\d+)$")

_CLEF_SIGN_REF_DIATONIC: Dict[str, int] = {
    "G": 4 * 7 + STEP_TO_IDX["G"],  # G4 = 32
    "F": 3 * 7 + STEP_TO_IDX["F"],  # F3 = 24
    "C": 4 * 7 + STEP_TO_IDX["C"],  # C4 = 28
}
_CLEF_OCT_SHIFT: Dict[Optional[str], int] = {
    None: 0,
    "8vb": -7,
    "8va": 7,
    "15mb": -14,
    "15ma": 14,
}


def diatonic_index(step: str, octave: int) -> int:
    return int(octave) * 7 + STEP_TO_IDX[step]


def from_diatonic_index(idx: int) -> Tuple[str, int]:
    octave, step_idx = divmod(int(idx), 7)
    return IDX_TO_STEP[step_idx], octave


def clef_bottom_line_diatonic(clef_tok: str) -> int:
    m = CLEF_PATTERN.match(clef_tok)
    if not m:
        return 30  # E4 (treble default)
    sign, line_s, oct_mod = m.group(1), m.group(2), m.group(3)
    ref = _CLEF_SIGN_REF_DIATONIC.get(sign)
    if ref is None:
        return 30
    return (ref + _CLEF_OCT_SHIFT.get(oct_mod, 0)) - 2 * (int(line_s) - 1)


def key_signature_alters(key_tok: str) -> Dict[str, int]:
    alters = {s: 0 for s in IDX_TO_STEP}
    m = KEY_PATTERN.match(key_tok)
    if not m:
        return alters
    fifths = int(m.group(1))
    if fifths > 0:
        for s in SHARP_ORDER[: min(7, fifths)]:
            alters[s] = 1
    elif fifths < 0:
        for s in FLAT_ORDER[: min(7, -fifths)]:
            alters[s] = -1
    return alters


def dotted_duration_fraction(base_dur: str, dots: int) -> Fraction:
    base = DUR_TO_FRACTION.get(base_dur, Fraction(1, 4))
    # Exact geometric sum: base * (2 - 2^-dots)
    mult = Fraction(2 ** (dots + 1) - 1, 2 ** dots)
    return base * mult


def fuse_duration(base_dur: str, dots: int) -> str:
    d_clamped = max(0, min(2, int(dots)))
    if d_clamped == 0:
        return base_dur
    return f"{base_dur}{'.' * d_clamped}"


def encode_int_digits(val: int) -> List[str]:
    return [CHAR_TO_DIGIT[ch] for ch in str(max(0, int(val)))]


def encode_sp_with_escape(sp: int) -> List[str]:
    out: List[str] = []
    cur = int(sp)
    while cur < SP_CORE_MIN:
        out.append(SP_OCT_DN)
        cur += 7
    while cur > SP_CORE_MAX:
        out.append(SP_OCT_UP)
        cur -= 7
    out.append(f"sp{cur:+d}")
    return out


@dataclass
class RatioGaugeState:
    """Exact Ratio<i64> (Fraction) pitch, accidental, and rhythmic state tracker."""
    clef: str = "clef-G-2"
    bottom_diat: int = 30  # E4
    key_alters: Dict[str, int] = field(default_factory=lambda: {s: 0 for s in IDX_TO_STEP})
    in_measure: Dict[Tuple[str, int], int] = field(default_factory=dict)
    measure_pos: Fraction = Fraction(0, 1)
    tuplet_stack: List[Fraction] = field(default_factory=list)
    meter: Fraction = Fraction(4, 4)

    def set_clef(self, clef_tok: str) -> None:
        self.clef = clef_tok
        self.bottom_diat = clef_bottom_line_diatonic(clef_tok)

    def set_key(self, key_tok: str) -> None:
        self.key_alters = key_signature_alters(key_tok)

    def set_meter(self, num: int, den: int) -> None:
        if num > 0 and den > 0:
            self.meter = Fraction(num, den)

    def reset_measure(self) -> None:
        self.in_measure.clear()
        self.measure_pos = Fraction(0, 1)
        self.tuplet_stack.clear()

    def reset_voice_cursor(self) -> None:
        self.measure_pos = Fraction(0, 1)
        self.tuplet_stack.clear()

    def active_alter(self, step: str, octave: int) -> int:
        if (step, octave) in self.in_measure:
            return self.in_measure[(step, octave)]
        return self.key_alters.get(step, 0)

    def record_alter(self, step: str, octave: int, alter: int) -> None:
        self.in_measure[(step, octave)] = alter

    def active_tuplet_mult(self) -> Fraction:
        m = Fraction(1, 1)
        for f in self.tuplet_stack:
            m *= f
        return m

    def advance_time(self, base_dur: str, dots: int) -> Fraction:
        dt = dotted_duration_fraction(base_dur, dots) * self.active_tuplet_mult()
        self.measure_pos += dt
        return dt


# ---------------------------------------------------------------------------
# bekern3_core Vocabulary Builder
# ---------------------------------------------------------------------------

def build_bekern3_core_vocab(
    base_vocab: Sequence[str],
    sys_token: str = "<sys>",
) -> Tuple[List[str], Dict[str, int], Dict[int, str], Dict[str, int]]:
    """Build compact, OOV-proof `bekern3_core` vocabulary from `bekern+v2` base_vocab."""
    removed = STEPS_SET | OCTAVES_SET | frozenset(ALTER_TO_INT.keys()) | frozenset(("dot", "tuplet-stop"))
    kept: List[str] = []
    for t in base_vocab:
        if t in removed:
            continue
        if TIME_V2_PATTERN.match(t) or TUPLET_V2_PATTERN.match(t) or VOICE_V2_PATTERN.match(t):
            continue
        kept.append(t)

    added: List[str] = (
        SP_CORE_TOKENS
        + [SP_OCT_DN, SP_OCT_UP]
        + ACC_TOKENS
        + FUSED_DUR_TOKENS
        + [VOICE_SEP, SKIP_TOK, TIME_PREFIX, TUP_PREFIX, TUP_END, RAT_SLASH]
        + list(DIGIT_TOKENS)
    )
    b3_base = list(dict.fromkeys(kept + added))
    pad_id = len(b3_base)
    bos_id = len(b3_base) + 1
    eos_id = len(b3_base) + 2
    sys_id = len(b3_base) + 3
    vocab_size = len(b3_base) + 4

    tok_to_id: Dict[str, int] = {t: i for i, t in enumerate(b3_base)}
    tok_to_id[sys_token] = sys_id
    id_to_tok: Dict[int, str] = {i: t for i, t in enumerate(b3_base)}
    id_to_tok[pad_id] = "<pad>"
    id_to_tok[bos_id] = "<bos>"
    id_to_tok[eos_id] = "<eos>"
    id_to_tok[sys_id] = sys_token
    specials = {
        "pad_id": pad_id,
        "bos_id": bos_id,
        "eos_id": eos_id,
        "sys_id": sys_id,
        "vocab_size": vocab_size,
    }
    return b3_base, tok_to_id, id_to_tok, specials


# ---------------------------------------------------------------------------
# Lossless bekern+v2 <-> bekern3_core Compiler & Exact Ratio<i64> Resolver
# ---------------------------------------------------------------------------

def _compile_voice_segment_to_b3(
    seg: Sequence[str],
    st: RatioGaugeState,
) -> Tuple[List[str], List[int]]:
    """Compile a single voice segment of `bekern+v2` tokens into `bekern3_core`.

    Returns `(b3_tokens, list_of_sp_values)` where `list_of_sp_values` is used
    to compute exact `mean(sp)` for descending-`mean(sp)` voice canonicalization.
    """
    out: List[str] = []
    sps: List[int] = []
    n = len(seg)
    i = 0
    active_chord_dur: Optional[str] = None

    while i < n:
        tok = seg[i]

        if tok.startswith("clef-"):
            st.set_clef(tok)
            out.append(tok)
            active_chord_dur = None
            i += 1
            continue

        if tok.startswith("key") and KEY_PATTERN.match(tok):
            st.set_key(tok)
            out.append(tok)
            active_chord_dur = None
            i += 1
            continue

        m_time = TIME_V2_PATTERN.match(tok)
        if m_time:
            num, den = int(m_time.group(1)), int(m_time.group(2))
            st.set_meter(num, den)
            out.append(TIME_PREFIX)
            out.extend(encode_int_digits(num))
            out.append(RAT_SLASH)
            out.extend(encode_int_digits(den))
            active_chord_dur = None
            i += 1
            continue

        m_tup = TUPLET_V2_PATTERN.match(tok)
        if m_tup:
            actual, normal = int(m_tup.group(1)), int(m_tup.group(2))
            if actual > 0 and normal > 0:
                st.tuplet_stack.append(Fraction(normal, actual))
            out.append(TUP_PREFIX)
            out.extend(encode_int_digits(actual))
            out.append(RAT_SLASH)
            out.extend(encode_int_digits(normal))
            active_chord_dur = None
            i += 1
            continue

        if tok == "tuplet-stop":
            if st.tuplet_stack:
                st.tuplet_stack.pop()
            out.append(TUP_END)
            active_chord_dur = None
            i += 1
            continue

        if tok == "rest":
            active_chord_dur = None
            if (i + 1) < n and seg[i + 1] in BASE_DURATIONS_SET:
                base_dur = seg[i + 1]
                j = i + 2
                dots = 0
                while j < n and seg[j] == "dot":
                    dots += 1
                    j += 1
                fused = fuse_duration(base_dur, dots)
                st.advance_time(base_dur, min(2, dots))
                out.append("rest")
                out.append(fused)
                i = j
                continue
            out.append(tok)
            i += 1
            continue

        # Chord tone: `chord <DUR> <STEP> <OCT> [a*] [dot]*`
        if tok == "chord" and (i + 3) < n and seg[i + 1] in BASE_DURATIONS_SET and seg[i + 2] in STEPS_SET and seg[i + 3] in OCTAVES_SET:
            base_dur = seg[i + 1]
            step = seg[i + 2]
            octave = int(seg[i + 3])
            j = i + 4
            alter = 0
            if j < n and seg[j] in ALTER_TO_INT:
                alter = ALTER_TO_INT[seg[j]]
                j += 1
            dots = 0
            while j < n and seg[j] == "dot":
                dots += 1
                j += 1
            fused = fuse_duration(base_dur, dots)
            # Implicit chord when duration matches chord root; explicit fallback if mixed duration
            if active_chord_dur is None or fused != active_chord_dur:
                out.append("chord")
                out.append(fused)
            sp = diatonic_index(step, octave) - st.bottom_diat
            sps.append(sp)
            out.extend(encode_sp_with_escape(sp))
            if alter != st.active_alter(step, octave):
                out.append(INT_TO_ACC[alter])
                st.record_alter(step, octave, alter)
            i = j
            continue

        # Standard note onset: `<DUR> <STEP> <OCT> [a*] [dot]*`
        if tok in BASE_DURATIONS_SET and (i + 2) < n and seg[i + 1] in STEPS_SET and seg[i + 2] in OCTAVES_SET:
            base_dur = tok
            step = seg[i + 1]
            octave = int(seg[i + 2])
            j = i + 3
            alter = 0
            if j < n and seg[j] in ALTER_TO_INT:
                alter = ALTER_TO_INT[seg[j]]
                j += 1
            dots = 0
            while j < n and seg[j] == "dot":
                dots += 1
                j += 1
            fused = fuse_duration(base_dur, dots)
            st.advance_time(base_dur, min(2, dots))
            active_chord_dur = fused
            out.append(fused)
            sp = diatonic_index(step, octave) - st.bottom_diat
            sps.append(sp)
            out.extend(encode_sp_with_escape(sp))
            if alter != st.active_alter(step, octave):
                out.append(INT_TO_ACC[alter])
                st.record_alter(step, octave, alter)
            i = j
            continue

        # Stray pitch without preceding duration (defensive)
        if tok in STEPS_SET and (i + 1) < n and seg[i + 1] in OCTAVES_SET:
            step = tok
            octave = int(seg[i + 1])
            j = i + 2
            alter = 0
            if j < n and seg[j] in ALTER_TO_INT:
                alter = ALTER_TO_INT[seg[j]]
                j += 1
            sp = diatonic_index(step, octave) - st.bottom_diat
            sps.append(sp)
            out.extend(encode_sp_with_escape(sp))
            if alter != st.active_alter(step, octave):
                out.append(INT_TO_ACC[alter])
                st.record_alter(step, octave, alter)
            i = j
            continue

        if tok not in POST_NOTE_MARKS:
            active_chord_dur = None
        out.append(tok)
        i += 1

    return out, sps


def bekern_v2a_to_bekern3_core(
    tokens: Sequence[str],
    sys_token: str = "<sys>",
) -> List[str]:
    """Compile `bekern+v2` Phase A tokens into `bekern3_core`."""
    st = RatioGaugeState()
    out: List[str] = []

    def _flush_measure(m_toks: List[str]) -> None:
        if not m_toks:
            return
        # Check whether measure has explicit voice-N segments
        has_voice = any(VOICE_V2_PATTERN.match(t) for t in m_toks)
        if not has_voice:
            compiled, _ = _compile_voice_segment_to_b3(m_toks, st)
            out.extend(compiled)
            return

        # Split into prefix (before first voice-N) + voice segments
        prefix: List[str] = []
        voices: List[List[str]] = []
        cur_v: Optional[List[str]] = None
        for t in m_toks:
            if VOICE_V2_PATTERN.match(t):
                if cur_v is not None:
                    voices.append(cur_v)
                cur_v = []
            elif cur_v is None:
                prefix.append(t)
            else:
                cur_v.append(t)
        if cur_v is not None:
            voices.append(cur_v)

        if prefix:
            comp_pre, _ = _compile_voice_segment_to_b3(prefix, st)
            out.extend(comp_pre)

        # Compile each voice segment with snapshot of measure gauge, then sort by descending mean(sp)
        compiled_voices: List[Tuple[float, int, List[str]]] = []
        base_clef = st.clef
        base_bottom = st.bottom_diat
        base_key = dict(st.key_alters)
        for v_idx, v_seg in enumerate(voices):
            v_st = RatioGaugeState(
                clef=base_clef,
                bottom_diat=base_bottom,
                key_alters=dict(base_key),
                in_measure={},
                meter=st.meter,
            )
            comp_v, sps = _compile_voice_segment_to_b3(v_seg, v_st)
            mean_sp = (sum(sps) / float(len(sps))) if sps else -1e9
            compiled_voices.append((mean_sp, v_idx, comp_v))

        # Descending mean(sp), deterministic tie-break on original voice index
        compiled_voices.sort(key=lambda item: (-item[0], item[1]))
        for idx_sorted, (_, _, comp_v) in enumerate(compiled_voices):
            if idx_sorted > 0:
                out.append(VOICE_SEP)
            out.extend(comp_v)

    cur_measure: List[str] = []
    for tok in tokens:
        if tok == sys_token or tok == "<staff>" or tok in BAR_RESET_TOKENS:
            _flush_measure(cur_measure)
            cur_measure = []
            st.reset_measure()
            out.append(tok)
        else:
            cur_measure.append(tok)
    _flush_measure(cur_measure)
    return out


def _read_rational_pair(tokens: Sequence[str], start: int) -> Tuple[Optional[Tuple[int, int]], int]:
    """Read `<d0..d9>+ / <d0..d9>+` starting at `start`."""
    n = len(tokens)
    i = start
    num_chars: List[str] = []
    while i < n and tokens[i] in DIGIT_TO_CHAR:
        num_chars.append(DIGIT_TO_CHAR[tokens[i]])
        i += 1
    if not num_chars or i >= n or tokens[i] != RAT_SLASH:
        return None, start
    i += 1  # consume '/'
    den_chars: List[str] = []
    while i < n and tokens[i] in DIGIT_TO_CHAR:
        den_chars.append(DIGIT_TO_CHAR[tokens[i]])
        i += 1
    if not den_chars:
        return None, start
    return (int("".join(num_chars)), int("".join(den_chars))), i


def bekern3_core_to_bekern_v2a(
    tokens: Sequence[str],
    sys_token: str = "<sys>",
) -> Tuple[List[str], Dict[str, Any]]:
    """Zero-parameter deterministic Ratio<i64> resolver: `bekern3_core -> bekern+v2`."""
    st = RatioGaugeState()
    out: List[str] = []
    n = len(tokens)
    i = 0
    cur_dur: Optional[Tuple[str, int]] = None
    notes_in_current_onset = 0
    explicit_chord_pending = False
    voice_idx_in_measure = 1
    measures_checked = 0
    measures_exact_meter = 0
    total_notes_resolved = 0

    while i < n:
        tok = tokens[i]

        if tok == sys_token or tok == "<staff>" or tok in BAR_RESET_TOKENS:
            if st.measure_pos > 0:
                measures_checked += 1
                if st.measure_pos == st.meter:
                    measures_exact_meter += 1
            st.reset_measure()
            cur_dur = None
            notes_in_current_onset = 0
            explicit_chord_pending = False
            voice_idx_in_measure = 1
            out.append(tok)
            i += 1
            continue

        if tok == VOICE_SEP:
            voice_idx_in_measure += 1
            st.reset_voice_cursor()
            st.in_measure.clear()
            cur_dur = None
            notes_in_current_onset = 0
            explicit_chord_pending = False
            out.append(f"voice-{min(27, voice_idx_in_measure)}")
            i += 1
            continue

        if tok.startswith("clef-"):
            st.set_clef(tok)
            out.append(tok)
            cur_dur = None
            notes_in_current_onset = 0
            i += 1
            continue

        if tok.startswith("key") and KEY_PATTERN.match(tok):
            st.set_key(tok)
            out.append(tok)
            cur_dur = None
            notes_in_current_onset = 0
            i += 1
            continue

        if tok == TIME_PREFIX:
            pair, nxt = _read_rational_pair(tokens, i + 1)
            if pair is not None:
                st.set_meter(pair[0], pair[1])
                out.append(f"time-{pair[0]}-{pair[1]}")
                i = nxt
            else:
                i += 1
            cur_dur = None
            notes_in_current_onset = 0
            continue

        if tok == TUP_PREFIX:
            pair, nxt = _read_rational_pair(tokens, i + 1)
            if pair is not None:
                actual, normal = pair
                if actual > 0 and normal > 0:
                    st.tuplet_stack.append(Fraction(normal, actual))
                out.append(f"tuplet-start-{actual}-{normal}")
                i = nxt
            else:
                i += 1
            cur_dur = None
            notes_in_current_onset = 0
            continue

        if tok == TUP_END:
            if st.tuplet_stack:
                st.tuplet_stack.pop()
            out.append("tuplet-stop")
            cur_dur = None
            notes_in_current_onset = 0
            i += 1
            continue

        if tok in ("rest", SKIP_TOK):
            cur_dur = None
            notes_in_current_onset = 0
            explicit_chord_pending = False
            if (i + 1) < n and tokens[i + 1] in ALL_FUSED_DURS_SET:
                base_dur, dots = FUSED_DUR_PARSE[tokens[i + 1]]
                st.advance_time(base_dur, dots)
                out.append("rest")
                out.append(base_dur)
                for _ in range(dots):
                    out.append("dot")
                i += 2
            else:
                if tok == "rest":
                    out.append("rest")
                i += 1
            continue

        if tok == "chord":
            explicit_chord_pending = True
            i += 1
            continue

        if tok in ALL_FUSED_DURS_SET:
            base_dur, dots = FUSED_DUR_PARSE[tok]
            cur_dur = (base_dur, dots)
            if not explicit_chord_pending:
                notes_in_current_onset = 0
                st.advance_time(base_dur, dots)
            i += 1
            continue

        # Notehead: `[sp-oct-dn|sp-oct-up]* sp<k> [acc-*]`
        if tok in (SP_OCT_DN, SP_OCT_UP) or SP_PATTERN.match(tok):
            oct_shift = 0
            j = i
            while j < n and tokens[j] in (SP_OCT_DN, SP_OCT_UP):
                oct_shift += -7 if tokens[j] == SP_OCT_DN else 7
                j += 1
            if j < n:
                m_sp = SP_PATTERN.match(tokens[j])
            else:
                m_sp = None
            if m_sp is None:
                i = max(i + 1, j)
                continue
            sp_val = int(m_sp.group(1)) + oct_shift
            j += 1
            step, octave = from_diatonic_index(st.bottom_diat + sp_val)
            if j < n and tokens[j] in ACC_TO_INT:
                alter = ACC_TO_INT[tokens[j]]
                st.record_alter(step, octave, alter)
                j += 1
            else:
                alter = st.active_alter(step, octave)

            base_dur, dots = cur_dur if cur_dur is not None else ("quarter", 0)
            is_chord_tone = explicit_chord_pending or (notes_in_current_onset > 0)
            if is_chord_tone:
                out.append("chord")
            out.append(base_dur)
            out.append(step)
            out.append(str(octave))
            if alter != 0:
                out.append(INT_TO_ALTER[alter])
            for _ in range(dots):
                out.append("dot")

            notes_in_current_onset += 1
            explicit_chord_pending = False
            total_notes_resolved += 1
            i = j
            continue

        # Defensive skip of stray accidental or rational digit tokens
        if tok in ACC_TO_INT or tok in DIGIT_TO_CHAR or tok == RAT_SLASH:
            i += 1
            continue

        if tok not in POST_NOTE_MARKS:
            cur_dur = None
            notes_in_current_onset = 0
            explicit_chord_pending = False
        out.append(tok)
        i += 1

    stats = {
        "measures_checked": measures_checked,
        "measures_exact_meter": measures_exact_meter,
        "meter_exact_rate": round(measures_exact_meter / max(1, measures_checked), 4),
        "total_notes_resolved": total_notes_resolved,
    }
    return out, stats


# ---------------------------------------------------------------------------
# MusicXML -> bekern+v2 Extraction & Suite31 Oracle Helpers
# ---------------------------------------------------------------------------

def _strip_xml_tag(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _musicxml_to_bekern_v2a_and_notes(
    xml_path: Path,
) -> Tuple[List[str], List[Tuple[str, int, str, int, int]], bool]:
    """Parse MusicXML into (bekern_v2a_tokens, [(dur, dots, step, oct, alter)...], is_cclef_or_key_heavy)."""
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    toks: List[str] = []
    notes: List[Tuple[str, int, str, int, int]] = []
    is_heavy = False

    for elem in root.iter():
        tag = _strip_xml_tag(elem.tag)
        if tag == "measure":
            if toks and toks[-1] != "bar":
                toks.append("bar")
            for ch in elem:
                ctag = _strip_xml_tag(ch.tag)
                if ctag == "attributes":
                    for ach in ch:
                        atag = _strip_xml_tag(ach.tag)
                        if atag == "key":
                            for kch in ach:
                                if _strip_xml_tag(kch.tag) == "fifths" and kch.text:
                                    try:
                                        f5 = int(kch.text.strip())
                                        toks.append(f"key{f5:+d}")
                                        if abs(f5) >= 3:
                                            is_heavy = True
                                    except ValueError:
                                        pass
                        elif atag == "time":
                            beats, btype = "4", "4"
                            for tch in ach:
                                ttag = _strip_xml_tag(tch.tag)
                                if ttag == "beats" and tch.text:
                                    beats = tch.text.strip()
                                elif ttag == "beat-type" and tch.text:
                                    btype = tch.text.strip()
                            if beats.isdigit() and btype.isdigit():
                                toks.append(f"time-{beats}-{btype}")
                        elif atag == "clef":
                            sign, line, oct_chg = "G", "2", 0
                            for cch in ach:
                                cltag = _strip_xml_tag(cch.tag)
                                if cltag == "sign" and cch.text:
                                    sign = cch.text.strip().upper()
                                elif cltag == "line" and cch.text:
                                    line = cch.text.strip()
                                elif cltag == "clef-octave-change" and cch.text:
                                    try:
                                        oct_chg = int(cch.text.strip())
                                    except ValueError:
                                        oct_chg = 0
                            if sign in ("C", "F"):
                                is_heavy = True
                            suf = "-8vb" if oct_chg == -1 else ("-8va" if oct_chg == 1 else "")
                            toks.append(f"clef-{sign}-{line}{suf}")
                elif ctag == "note":
                    is_rest = False
                    is_grace = False
                    is_chord = False
                    step = ""
                    octave = 4
                    alter = 0
                    dur = "quarter"
                    dots = 0
                    for nch in ch:
                        ntag = _strip_xml_tag(nch.tag)
                        if ntag == "rest":
                            is_rest = True
                        elif ntag == "grace":
                            is_grace = True
                        elif ntag == "chord":
                            is_chord = True
                        elif ntag == "pitch":
                            for pch in nch:
                                ptag = _strip_xml_tag(pch.tag)
                                if ptag == "step" and pch.text:
                                    step = pch.text.strip().upper()
                                elif ptag == "octave" and pch.text:
                                    try:
                                        octave = int(pch.text.strip())
                                    except ValueError:
                                        octave = 4
                                elif ptag == "alter" and pch.text:
                                    try:
                                        alter = int(round(float(pch.text.strip())))
                                    except ValueError:
                                        alter = 0
                        elif ntag == "type" and nch.text:
                            raw_d = nch.text.strip().lower()
                            if raw_d in BASE_DURATIONS_SET:
                                dur = raw_d
                        elif ntag == "dot":
                            dots += 1
                    dots_c = min(2, dots)
                    if is_grace:
                        continue
                    if is_rest:
                        toks.extend(["rest", dur] + ["dot"] * dots_c)
                    elif step:
                        if is_chord:
                            toks.append("chord")
                        toks.extend([dur, step, str(octave)])
                        if alter in INT_TO_ALTER:
                            toks.append(INT_TO_ALTER[alter])
                        toks.extend(["dot"] * dots_c)
                        notes.append((dur, dots_c, step, octave, alter))
    return toks, notes, is_heavy


def _extract_bekern_notes(toks: Sequence[str]) -> List[Tuple[str, int, str, int, int]]:
    notes: List[Tuple[str, int, str, int, int]] = []
    cur_dur = "quarter"
    in_rest = False
    i = 0
    n = len(toks)
    while i < n:
        t = toks[i]
        if t == "rest":
            in_rest = True
            i += 1
            continue
        if t in BASE_DURATIONS_SET:
            if in_rest:
                in_rest = False
            else:
                cur_dur = t
            i += 1
            continue
        if t in STEPS_SET and (i + 1) < n and toks[i + 1] in OCTAVES_SET:
            step = t
            octv = int(toks[i + 1])
            i += 2
            alt = 0
            if i < n and toks[i] in ALTER_TO_INT:
                alt = ALTER_TO_INT[toks[i]]
                i += 1
            dots = 0
            while i < n and toks[i] == "dot":
                dots += 1
                i += 1
            notes.append((cur_dur, min(2, dots), step, octv, alt))
            continue
        i += 1
    return notes


def _multiset_f1(ref: Sequence[Any], hyp: Sequence[Any]) -> float:
    if not ref and not hyp:
        return 1.0
    if not ref or not hyp:
        return 0.0
    rc = Counter(ref)
    hc = Counter(hyp)
    inter = sum((rc & hc).values())
    return float(2 * inter) / float(len(ref) + len(hyp))


# ---------------------------------------------------------------------------
# Infrastructure Helpers (HF Hub, Dataset Prep, Atomic IO)
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _get_vol_root() -> Path:
    if Path("/vol").is_dir():
        return Path("/vol")
    if Path("/content").is_dir():
        return Path("/content/vol")
    return Path("/tmp/omr_vol")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_hf_token() -> str:
    tok = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    if not tok and HF_TOKEN_FILE.is_file():
        try:
            os.chmod(HF_TOKEN_FILE, 0o600)
        except OSError:
            pass
        tok = HF_TOKEN_FILE.read_text(encoding="utf-8").strip()
    if HF_TOKEN_FILE.exists():
        try:
            HF_TOKEN_FILE.unlink()
        except OSError:
            pass
    if not tok:
        raise RuntimeError("HF token missing: set HF_TOKEN env var or /tmp/.hf_token")
    return tok


def _ensure_deps() -> None:
    needed = []
    for mod, pkg in (
        ("huggingface_hub", "huggingface_hub>=0.26"),
        ("PIL", "pillow"),
        ("numpy", "numpy"),
    ):
        try:
            __import__(mod)
        except ImportError:
            needed.append(pkg)
    if needed:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *needed])


def _locate_mvd() -> Path:
    for cand in (
        Path("/workspace/mvd"),
        Path("/workspace/sheet-music-converter/tmp-train/mvd"),
        Path("/content/mvd"),
        Path("/content/sheet-music-converter/tmp-train/mvd"),
        Path.cwd() / "tmp-train" / "mvd",
        Path(__file__).resolve().parents[2] / "tmp-train" / "mvd",
    ):
        if (cand / "uni_omr_v2.py").is_file() and (cand / "data.py").is_file():
            return cand
    for bundle in (Path("/content/bundle.tar.gz"), Path("/content/repo_bundle.tar.gz")):
        if bundle.is_file():
            dest = Path("/content/sheet-music-converter")
            dest.mkdir(parents=True, exist_ok=True)
            with tarfile.open(bundle, "r:*") as tf:
                tf.extractall(path=dest)
            mvd = dest / "tmp-train" / "mvd"
            if (mvd / "uni_omr_v2.py").is_file():
                return mvd
    raise FileNotFoundError("Could not locate tmp-train/mvd containing uni_omr_v2.py and data.py")


def _ensure_proxy_dataset(vol_root: Path, hf_token: str) -> Tuple[Path, Path]:
    from huggingface_hub import hf_hub_download

    ds_dir = vol_root / "datasets" / "iter10_5_proxy_2k"
    train_abs = ds_dir / "proxy_train_abs.jsonl"
    val_abs = ds_dir / "proxy_val_abs.jsonl"
    if train_abs.is_file() and val_abs.is_file():
        return train_abs, val_abs

    ds_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{EXP_ID}] Downloading {HF_PROXY_TAR} from {HF_REPO_ID}...", flush=True)
    tar_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_PROXY_TAR,
        repo_type="dataset",
        token=hf_token,
        local_dir=str(vol_root / "tars"),
    )
    with tarfile.open(tar_path, "r:*") as tf:
        tf.extractall(path=ds_dir)
    Path(tar_path).unlink(missing_ok=True)

    for src_name, dst_path in (("proxy_train.jsonl", train_abs), ("proxy_val.jsonl", val_abs)):
        src_path = ds_dir / src_name
        out_lines: List[str] = []
        for raw in src_path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            rec = json.loads(raw)
            rec["image"] = str((ds_dir / rec["image"]).resolve())
            rec["bekerns"] = [str((ds_dir / b).resolve()) for b in rec.get("bekerns", [])]
            out_lines.append(json.dumps(rec))
        dst_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return train_abs, val_abs


def _ensure_suite31(vol_root: Path, hf_token: str) -> Path:
    from huggingface_hub import hf_hub_download

    s31_dir = vol_root / "datasets" / "suite31_hw_eval"
    sentinel = s31_dir / ".extracted"
    if not sentinel.exists():
        print(f"[{EXP_ID}] Downloading {HF_SUITE31_TAR} from {HF_REPO_ID}...", flush=True)
        tar_p = Path(
            hf_hub_download(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                filename=HF_SUITE31_TAR,
                local_dir=str(vol_root / "tars"),
                token=hf_token,
            )
        )
        s31_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_p, "r:*") as tf:
            tf.extractall(path=s31_dir)
        sentinel.write_text(_utc_now() + "\n", encoding="utf-8")
        tar_p.unlink(missing_ok=True)
    manifests = sorted(s31_dir.rglob("suite31_hw_manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"Missing suite31_hw_manifest.json under {s31_dir}")
    return manifests[0]


def _upload_json(hf_token: str, local_path: Path, path_in_repo: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        commit_message=f"{EXP_ID}: upload {path_in_repo}",
    )
    print(f"[{EXP_ID}] Uploaded {path_in_repo} -> {HF_REPO_ID}", flush=True)


def _load_manifest_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _read_page_tokens(rec: Dict[str, Any], sys_token: str) -> List[str]:
    if "tokens" in rec and isinstance(rec["tokens"], list):
        return [str(t) for t in rec["tokens"]]
    chunks: List[List[str]] = []
    for bp in rec.get("bekerns", []):
        p = Path(bp)
        if p.is_file():
            toks = p.read_text(encoding="utf-8").split()
            if toks:
                chunks.append(toks)
    out: List[str] = []
    for idx, ch in enumerate(chunks):
        if idx > 0:
            out.append(sys_token)
        out.extend(ch)
    return out


def _load_rgb_tensor(img_path: Path, max_dim: int, uni_omr_mod: Any) -> Any:
    from PIL import Image

    with Image.open(img_path) as im:
        pil_im = uni_omr_mod._resize_fullpage_pil(im.convert("RGB"), max_dim)
    return uni_omr_mod._pil_to_chw_tensor(pil_im).unsqueeze(0)


# ---------------------------------------------------------------------------
# Phase 0: Deterministic Suite31 Oracle & Compression Audit
# ---------------------------------------------------------------------------

def _run_oracle_and_compression_audit(
    train_rows: Sequence[Dict[str, Any]],
    val_rows: Sequence[Dict[str, Any]],
    s31_pieces: Sequence[Dict[str, Any]],
    sys_token: str,
) -> Dict[str, Any]:
    s31_f1s: List[float] = []
    s31_len_v2a: List[int] = []
    s31_len_b3: List[int] = []
    s31_meter_exact = 0
    s31_meter_total = 0

    for piece in s31_pieces:
        gt_p = Path(piece["gt"])
        if not gt_p.is_file():
            continue
        v2a_toks, gt_notes, _ = _musicxml_to_bekern_v2a_and_notes(gt_p)
        if not v2a_toks or not gt_notes:
            continue
        b3_toks = bekern_v2a_to_bekern3_core(v2a_toks, sys_token=sys_token)
        rt_v2a, r_stats = bekern3_core_to_bekern_v2a(b3_toks, sys_token=sys_token)
        rt_notes = _extract_bekern_notes(rt_v2a)
        s31_f1s.append(_multiset_f1(gt_notes, rt_notes))
        s31_len_v2a.append(len(v2a_toks))
        s31_len_b3.append(len(b3_toks))
        s31_meter_exact += int(r_stats["measures_exact_meter"])
        s31_meter_total += int(r_stats["measures_checked"])

    def _corpus_stats(rows: Sequence[Dict[str, Any]], limit: int = 300) -> Dict[str, Any]:
        l_v2a: List[int] = []
        l_b3: List[int] = []
        rt_note_f1s: List[float] = []
        for rec in rows[:limit]:
            toks = _read_page_tokens(rec, sys_token)
            if not toks:
                continue
            b3 = bekern_v2a_to_bekern3_core(toks, sys_token=sys_token)
            rt, _ = bekern3_core_to_bekern_v2a(b3, sys_token=sys_token)
            l_v2a.append(len(toks))
            l_b3.append(len(b3))
            rt_note_f1s.append(_multiset_f1(_extract_bekern_notes(toks), _extract_bekern_notes(rt)))
        sum_v2a = max(1, sum(l_v2a))
        sum_b3 = sum(l_b3)
        return {
            "n_pages": len(l_v2a),
            "mean_len_v2a": round(sum_v2a / max(1, len(l_v2a)), 1),
            "mean_len_b3": round(sum_b3 / max(1, len(l_b3)), 1),
            "token_reduction_pct": round(100.0 * (1.0 - float(sum_b3) / float(sum_v2a)), 2),
            "note_roundtrip_f1": round(sum(rt_note_f1s) / max(1, len(rt_note_f1s)), 5),
        }

    s31_sum_v2a = max(1, sum(s31_len_v2a))
    s31_sum_b3 = sum(s31_len_b3)
    oracle_f1 = round(sum(s31_f1s) / max(1, len(s31_f1s)), 5)
    s31_red_pct = round(100.0 * (1.0 - float(s31_sum_b3) / float(s31_sum_v2a)), 2)

    return {
        "suite31_hw_oracle": {
            "n_pieces": len(s31_f1s),
            "roundtrip_note_f1": oracle_f1,
            "passes_99_5_gate": oracle_f1 >= 0.995,
            "mean_len_v2a": round(s31_sum_v2a / max(1, len(s31_len_v2a)), 1),
            "mean_len_b3": round(s31_sum_b3 / max(1, len(s31_len_b3)), 1),
            "token_reduction_pct": s31_red_pct,
            "passes_30pct_gate": s31_red_pct >= 30.0,
            "meter_exact_rate": round(s31_meter_exact / max(1, s31_meter_total), 4),
        },
        "proxy_train_sample": _corpus_stats(train_rows, limit=400),
        "proxy_val_sample": _corpus_stats(val_rows, limit=200),
    }


# ---------------------------------------------------------------------------
# Decoding: Greedy + MTP-4 Speculative Decoding Throughput Benchmark
# ---------------------------------------------------------------------------

def _truncate_past_kvs(past_kvs: Optional[List[Any]], keep_len: int) -> Optional[List[Any]]:
    if past_kvs is None:
        return None
    trimmed: List[Any] = []
    for entry in past_kvs:
        if isinstance(entry, tuple) and len(entry) >= 2:
            # Self-attn KV is (k, v, ...); k/v have shape (B, n_kv, T, head_dim)
            new_items = []
            for idx, item in enumerate(entry):
                if idx < 2 and hasattr(item, "shape") and item.ndim == 4:
                    new_items.append(item[:, :, :keep_len, :].contiguous())
                else:
                    new_items.append(item)
            trimmed.append(tuple(new_items))
        else:
            trimmed.append(entry)
    return trimmed


def _greedy_decode_custom(
    model: Any,
    img_t: Any,
    *,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_len: int,
    device: Any,
    amp_dtype: Optional[Any],
) -> List[int]:
    import torch

    ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if (amp_dtype is not None and device.type == "cuda") else contextlib.nullcontext()
    with torch.no_grad(), ctx:
        mem, _, _ = model.encode_image(img_t, return_aux=False)
        past_kvs = model.precompute_cross_kv(mem)
        cur_ids = torch.full((1, 1), bos_id, dtype=torch.long, device=device)
        out_ids: List[int] = []
        for _ in range(max_len):
            logits, _, past_kvs = model.decode_tokens(
                cur_ids,
                memory=None,
                return_mtp=False,
                return_aux=False,
                past_kvs=past_kvs,
                use_cache=True,
                apply_tok_dropout=False,
            )
            nxt = int(logits[0, -1, :].argmax(dim=-1).item())
            if nxt == eos_id or nxt == pad_id:
                break
            out_ids.append(nxt)
            cur_ids = torch.tensor([[nxt]], dtype=torch.long, device=device)
    return out_ids


def _mtp4_speculative_decode(
    model: Any,
    img_t: Any,
    *,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_len: int,
    device: Any,
    amp_dtype: Optional[Any],
) -> Tuple[List[int], Dict[str, Any]]:
    """Exact MTP-4 speculative decoding with KV-cache rollback, returning (ids, perf_stats)."""
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if (amp_dtype is not None and device.type == "cuda") else contextlib.nullcontext()
    out_ids: List[int] = []
    n_steps = 0
    total_accepted = 0

    with torch.no_grad(), ctx:
        mem, _, _ = model.encode_image(img_t, return_aux=False)
        past_kvs = model.precompute_cross_kv(mem)
        # Prime with BOS
        bos_t = torch.full((1, 1), bos_id, dtype=torch.long, device=device)
        logits, mtp_logits, past_kvs = model.decode_tokens(
            bos_t,
            memory=None,
            return_mtp=True,
            return_aux=False,
            past_kvs=past_kvs,
            use_cache=True,
            apply_tok_dropout=False,
        )
        n_steps += 1
        next_tok = int(logits[0, -1, :].argmax(dim=-1).item())
        if next_tok != eos_id and next_tok != pad_id:
            out_ids.append(next_tok)
            total_accepted += 1
            draft = [int(ml[0, -1, :].argmax(dim=-1).item()) for ml in mtp_logits]

            while len(out_ids) < max_len:
                # Verify [next_tok, draft_0, ..., draft_{k-1}] in one forward pass
                k = len(draft)
                cand = [next_tok] + draft
                cand_t = torch.tensor([cand], dtype=torch.long, device=device)
                cache_len_before = len(out_ids)  # BOS + out_ids[:-1] already in KV cache
                v_logits, v_mtp, past_kvs_new = model.decode_tokens(
                    cand_t,
                    memory=None,
                    return_mtp=True,
                    return_aux=False,
                    past_kvs=past_kvs,
                    use_cache=True,
                    apply_tok_dropout=False,
                )
                n_steps += 1
                preds = v_logits[0].argmax(dim=-1).tolist()  # length k+1

                # Match drafted tokens against verified argmax predictions
                accepted_draft = 0
                stopped = False
                for j in range(k):
                    ver_tok = int(preds[j])
                    if ver_tok == eos_id or ver_tok == pad_id:
                        stopped = True
                        break
                    if ver_tok == draft[j]:
                        out_ids.append(ver_tok)
                        accepted_draft += 1
                        if len(out_ids) >= max_len:
                            stopped = True
                            break
                    else:
                        out_ids.append(ver_tok)
                        next_tok = ver_tok
                        # Roll back KV cache to cache_len_before + 1 + j
                        past_kvs = _truncate_past_kvs(past_kvs_new, cache_len_before + 1 + j)
                        draft = [int(ml[0, j, :].argmax(dim=-1).item()) for ml in v_mtp]
                        total_accepted += accepted_draft + 1
                        break
                else:
                    # All k draft tokens accepted! Bonus token from preds[k]
                    bonus = int(preds[k])
                    if bonus == eos_id or bonus == pad_id or len(out_ids) >= max_len:
                        stopped = True
                        total_accepted += accepted_draft
                    else:
                        out_ids.append(bonus)
                        next_tok = bonus
                        past_kvs = past_kvs_new
                        draft = [int(ml[0, k, :].argmax(dim=-1).item()) for ml in v_mtp]
                        total_accepted += accepted_draft + 1

                if stopped:
                    break

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = max(1e-6, time.perf_counter() - t0)
    return out_ids, {
        "n_tokens": len(out_ids),
        "n_steps": n_steps,
        "mean_accept_len": round(float(total_accepted) / max(1, n_steps), 3),
        "elapsed_sec": elapsed,
        "tokens_per_sec": round(float(len(out_ids)) / elapsed, 2),
    }


# ---------------------------------------------------------------------------
# Evaluation & Training Loops
# ---------------------------------------------------------------------------

def _evaluate_arm(
    model: Any,
    use_b3: bool,
    val_rows: Sequence[Dict[str, Any]],
    s31_pieces: Sequence[Dict[str, Any]],
    *,
    vocab_meta: Dict[str, Any],
    device: Any,
    amp_dtype: Optional[Any],
    uni_omr_mod: Any,
    eval_s31: Any,
    run_full_s31: bool = True,
) -> Dict[str, Any]:
    model.eval()
    bos_id = int(vocab_meta["bos_id"])
    eos_id = int(vocab_meta["eos_id"])
    pad_id = int(vocab_meta["pad_id"])
    id_to_tok = vocab_meta["id_to_tok"]
    sys_tok = uni_omr_mod.SYS_TOKEN

    val_tf1s: List[float] = []
    val_nf1s: List[float] = []
    for rec in val_rows[:VAL_LIMIT]:
        img_p = Path(rec["image"])
        if not img_p.is_file():
            continue
        gt_v2a = _read_page_tokens(rec, sys_tok)
        if not gt_v2a:
            continue
        img_t = _load_rgb_tensor(img_p, FULLPAGE_MAX_DIM, uni_omr_mod).to(device)
        pred_ids = _greedy_decode_custom(
            model, img_t, bos_id=bos_id, eos_id=eos_id, pad_id=pad_id,
            max_len=EVAL_MAX_LEN, device=device, amp_dtype=amp_dtype,
        )
        pred_raw = [id_to_tok[i] for i in pred_ids if i in id_to_tok]
        pred_v2a = bekern3_core_to_bekern_v2a(pred_raw, sys_token=sys_tok)[0] if use_b3 else pred_raw
        val_tf1s.append(float(eval_s31.compute_token_lcs_f1(gt_v2a, pred_v2a)))
        val_nf1s.append(_multiset_f1(_extract_bekern_notes(gt_v2a), _extract_bekern_notes(pred_v2a)))

    def _mean(xs: Sequence[float]) -> Optional[float]:
        return round(sum(xs) / len(xs), 4) if xs else None

    result: Dict[str, Any] = {
        "proxy_val": {
            "n_pages": len(val_nf1s),
            "token_f1": _mean(val_tf1s),
            "note_f1": _mean(val_nf1s),
        }
    }
    if not run_full_s31:
        return result

    s31_slices: Dict[str, List[float]] = {
        "pr_clean": [],
        "pr_photo": [],
        "hw_clean": [],
        "hw_photo": [],
        "all45_clean": [],
        "all45_photo": [],
        "overall": [],
        "cclef_key_heavy": [],
        "standard_treble": [],
    }
    mtp_total_notes = 0
    mtp_total_tokens = 0
    mtp_total_sec = 0.0
    mtp_accept_lens: List[float] = []
    mtp_bench_count = 0

    pieces_eval = s31_pieces[:SUITE31_LIMIT] if SUITE31_LIMIT > 0 else s31_pieces
    for piece in pieces_eval:
        gt_path = Path(piece["gt"])
        if not gt_path.is_file():
            continue
        _, gt_notes, is_heavy = _musicxml_to_bekern_v2a_and_notes(gt_path)
        if not gt_notes:
            continue
        is_hw = bool(piece.get("handwritten", False))
        for view in ("clean", "photo"):
            img_p = Path(piece[view])
            if not img_p.is_file():
                continue
            img_t = _load_rgb_tensor(img_p, FULLPAGE_MAX_DIM, uni_omr_mod).to(device)
            if mtp_bench_count < MTP_BENCH_PAGES and view == "clean":
                pred_ids, perf = _mtp4_speculative_decode(
                    model, img_t, bos_id=bos_id, eos_id=eos_id, pad_id=pad_id,
                    max_len=EVAL_MAX_LEN, device=device, amp_dtype=amp_dtype,
                )
                mtp_bench_count += 1
                mtp_total_tokens += int(perf["n_tokens"])
                mtp_total_sec += float(perf["elapsed_sec"])
                mtp_accept_lens.append(float(perf["mean_accept_len"]))
            else:
                pred_ids = _greedy_decode_custom(
                    model, img_t, bos_id=bos_id, eos_id=eos_id, pad_id=pad_id,
                    max_len=EVAL_MAX_LEN, device=device, amp_dtype=amp_dtype,
                )
            pred_raw = [id_to_tok[i] for i in pred_ids if i in id_to_tok]
            pred_v2a = bekern3_core_to_bekern_v2a(pred_raw, sys_token=sys_tok)[0] if use_b3 else pred_raw
            stitched, _ = eval_s31.stitch_uni_tokens(pred_v2a)
            pred_notes = _extract_bekern_notes(stitched)
            if mtp_bench_count <= MTP_BENCH_PAGES and view == "clean":
                mtp_total_notes += len(pred_notes)

            nf1 = _multiset_f1(gt_notes, pred_notes)
            s31_slices[f"{'hw' if is_hw else 'pr'}_{view}"].append(nf1)
            s31_slices[f"all45_{view}"].append(nf1)
            s31_slices["overall"].append(nf1)
            if is_heavy:
                s31_slices["cclef_key_heavy"].append(nf1)
            else:
                s31_slices["standard_treble"].append(nf1)

    result["suite31_hw"] = {k: {"n": len(v), "strict_note_f1": _mean(v)} for k, v in s31_slices.items()}
    result["mtp4_throughput"] = {
        "n_pages": mtp_bench_count,
        "total_notes": mtp_total_notes,
        "total_tokens": mtp_total_tokens,
        "total_sec": round(mtp_total_sec, 3),
        "effective_notes_per_sec": round(float(mtp_total_notes) / max(1e-6, mtp_total_sec), 2),
        "tokens_per_sec": round(float(mtp_total_tokens) / max(1e-6, mtp_total_sec), 2),
        "mean_accept_len": _mean(mtp_accept_lens),
    }
    return result


def _train_arm(
    arm_name: str,
    use_b3: bool,
    train_rows: Sequence[Dict[str, Any]],
    val_rows: Sequence[Dict[str, Any]],
    s31_pieces: Sequence[Dict[str, Any]],
    out_dir: Path,
    *,
    vocab_meta: Dict[str, Any],
    device: Any,
    amp_dtype: Optional[Any],
    uni_omr_mod: Any,
    uni_omr_v2: Any,
    eval_s31: Any,
) -> Dict[str, Any]:
    import torch
    import torch.nn.functional as F

    arm_dir = out_dir / arm_name
    done_file = arm_dir / "arm_done.json"
    ckpt_file = arm_dir / "ckpt_latest.pt"
    if done_file.is_file():
        print(f"[{EXP_ID}] Resuming completed arm {arm_name} from {done_file}", flush=True)
        return json.loads(done_file.read_text(encoding="utf-8"))

    arm_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    random.seed(SEED)

    vocab_size = int(vocab_meta["vocab_size"])
    pad_id = int(vocab_meta["pad_id"])
    bos_id = int(vocab_meta["bos_id"])
    eos_id = int(vocab_meta["eos_id"])
    tok_to_id = vocab_meta["tok_to_id"]
    sys_tok = uni_omr_mod.SYS_TOKEN

    cfg = uni_omr_v2.UniOMRv2Config.small_45m()
    cfg.vocab_size = vocab_size
    cfg.tok_dropout = 0.0
    model = uni_omr_v2.UniOMRv2Net(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    start_step = 0
    ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if ckpt_file.is_file():
        st = torch.load(ckpt_file, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        ema_state = st["ema"]
        opt.load_state_dict(st["opt"])
        start_step = int(st.get("step", 0))
        print(f"[{EXP_ID}] Resumed {arm_name} at step {start_step}/{STEPS_PER_ARM}", flush=True)

    t0 = time.perf_counter()
    last_save = time.monotonic()
    losses: List[float] = []
    rng = random.Random(SEED + start_step)

    for step in range(start_step, STEPS_PER_ARM):
        model.train()
        opt.zero_grad(set_to_none=True)
        if step < WARMUP_STEPS:
            lr_now = LR * float(step + 1) / float(max(1, WARMUP_STEPS))
        else:
            prog = float(step - WARMUP_STEPS) / float(max(1, STEPS_PER_ARM - WARMUP_STEPS))
            lr_now = MIN_LR + 0.5 * (LR - MIN_LR) * (1.0 + math.cos(math.pi * min(1.0, prog)))
        for pg in opt.param_groups:
            pg["lr"] = lr_now

        step_loss = 0.0
        valid_micros = 0
        for _ in range(GRAD_ACCUM):
            rec = rng.choice(train_rows)
            img_p = Path(rec["image"])
            if not img_p.is_file():
                continue
            gt_v2a = _read_page_tokens(rec, sys_tok)
            if not gt_v2a:
                continue
            enc_toks = bekern_v2a_to_bekern3_core(gt_v2a, sys_token=sys_tok) if use_b3 else gt_v2a
            ids = [bos_id] + [tok_to_id[t] for t in enc_toks if t in tok_to_id][: MAX_SEQ_LEN - 2] + [eos_id]
            if len(ids) < 3:
                continue
            tgt_in = torch.tensor([ids[:-1]], device=device, dtype=torch.long)
            tgt_out = torch.tensor([ids[1:]], device=device, dtype=torch.long)
            img_t = _load_rgb_tensor(img_p, FULLPAGE_MAX_DIM, uni_omr_mod).to(device)

            ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if (amp_dtype is not None and device.type == "cuda") else contextlib.nullcontext()
            with ctx:
                out = model(img_t, tgt_in, return_aux=False, return_mtp=True)
                logits = out["logits"]
                ce_main = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    tgt_out.reshape(-1),
                    ignore_index=pad_id,
                )
                mtp_logits = out.get("mtp_logits") or []
                mtp_loss = ce_main.new_zeros(())
                mtp_n = 0
                for k_idx, ml in enumerate(mtp_logits, start=1):
                    if tgt_out.size(1) > k_idx:
                        mtp_loss = mtp_loss + F.cross_entropy(
                            ml[:, :-k_idx, :].reshape(-1, ml.size(-1)),
                            tgt_out[:, k_idx:].reshape(-1),
                            ignore_index=pad_id,
                        )
                        mtp_n += 1
                loss = ce_main + (MTP_LOSS_WEIGHT * (mtp_loss / float(mtp_n)) if mtp_n > 0 else 0.0)

            (loss / float(GRAD_ACCUM)).backward()
            step_loss += float(loss.detach().item()) / float(GRAD_ACCUM)
            valid_micros += 1

        if valid_micros == 0:
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.is_floating_point():
                    ema_state[k].mul_(GPU_EMA_DECAY).add_(v.detach(), alpha=1.0 - GPU_EMA_DECAY)
                else:
                    ema_state[k].copy_(v.detach())
        losses.append(step_loss)

        if (step + 1) % 100 == 0 or (step + 1) == STEPS_PER_ARM:
            print(f"[{EXP_ID}] {arm_name} step {step + 1}/{STEPS_PER_ARM} lr={lr_now:.2e} loss={step_loss:.4f}", flush=True)

        if (time.monotonic() - last_save) >= SAVE_EVERY_MINUTES * 60.0 or (step + 1) == STEPS_PER_ARM:
            tmp_ckpt = ckpt_file.with_suffix(".pt.tmp")
            torch.save(
                {"step": step + 1, "model": model.state_dict(), "ema": ema_state, "opt": opt.state_dict()},
                tmp_ckpt,
            )
            os.replace(tmp_ckpt, ckpt_file)
            last_save = time.monotonic()

    elapsed = time.perf_counter() - t0
    # Evaluate RAW on proxy_val first (short-proxy EMA lag check), then EMA on full suite31_hw + MTP-4
    raw_val = _evaluate_arm(
        model, use_b3, val_rows, s31_pieces,
        vocab_meta=vocab_meta, device=device, amp_dtype=amp_dtype,
        uni_omr_mod=uni_omr_mod, eval_s31=eval_s31, run_full_s31=False,
    )
    model.load_state_dict(ema_state)
    ema_eval = _evaluate_arm(
        model, use_b3, val_rows, s31_pieces,
        vocab_meta=vocab_meta, device=device, amp_dtype=amp_dtype,
        uni_omr_mod=uni_omr_mod, eval_s31=eval_s31, run_full_s31=True,
    )

    summary = {
        "arm": arm_name,
        "use_bekern3_core": use_b3,
        "vocab_size": vocab_size,
        "steps": STEPS_PER_ARM,
        "elapsed_sec": round(elapsed, 2),
        "final_25step_loss": round(sum(losses[-25:]) / max(1, len(losses[-25:])), 4) if losses else None,
        "raw_proxy_val": raw_val["proxy_val"],
        "ema_eval": ema_eval,
    }
    _atomic_write_json(done_file, summary)
    del model, opt, ema_state
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def main() -> None:
    started_at = _utc_now()
    hf_token = _read_hf_token()
    _ensure_deps()
    mvd_dir = _locate_mvd()
    if str(mvd_dir) not in sys.path:
        sys.path.insert(0, str(mvd_dir))

    import torch
    import data as mvd_data
    import eval_uni_suite31 as eval_s31
    import uni_omr
    import uni_omr_v2

    vol_root = _get_vol_root()
    out_dir = vol_root / "out" / EXP_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        cc_major = torch.cuda.get_device_capability(0)[0]
        amp_dtype: Optional[Any] = torch.bfloat16 if cc_major >= 8 else torch.float16
    else:
        amp_dtype = None

    train_manifest, val_manifest = _ensure_proxy_dataset(vol_root, hf_token)
    s31_manifest = _ensure_suite31(vol_root, hf_token)
    train_rows = _load_manifest_rows(train_manifest)
    val_rows = _load_manifest_rows(val_manifest)
    _, s31_pieces = eval_s31.resolve_manifest_pieces(s31_manifest)

    # Phase 0: Deterministic MusicXML round-trip oracle & compression audit
    audit = _run_oracle_and_compression_audit(train_rows, val_rows, s31_pieces, sys_token=uni_omr.SYS_TOKEN)
    print(f"[{EXP_ID}] Phase 0 Oracle & Compression Audit: {json.dumps(audit)}", flush=True)

    # Build both vocabularies
    b3_base, b3_tok_to_id, b3_id_to_tok, b3_spec = build_bekern3_core_vocab(
        mvd_data.VOCAB, sys_token=uni_omr.SYS_TOKEN,
    )
    vocab_v2a = {
        "vocab_size": int(uni_omr.VOCAB_SIZE),
        "pad_id": int(uni_omr.PAD_ID),
        "bos_id": int(uni_omr.BOS_ID),
        "eos_id": int(uni_omr.EOS_ID),
        "tok_to_id": uni_omr.UNI_TOKEN_TO_ID,
        "id_to_tok": uni_omr.UNI_ID_TO_TOKEN,
    }
    vocab_b3 = {
        "vocab_size": int(b3_spec["vocab_size"]),
        "pad_id": int(b3_spec["pad_id"]),
        "bos_id": int(b3_spec["bos_id"]),
        "eos_id": int(b3_spec["eos_id"]),
        "tok_to_id": b3_tok_to_id,
        "id_to_tok": b3_id_to_tok,
    }

    results: Dict[str, Any] = {}
    run_file = out_dir / f"{EXP_ID}.run.json"
    for arm_name, use_b3, v_meta in (
        ("arm0_bekern_v2a", False, vocab_v2a),
        ("arm1_bekern3_core", True, vocab_b3),
    ):
        results[arm_name] = _train_arm(
            arm_name, use_b3, train_rows, val_rows, s31_pieces, out_dir,
            vocab_meta=v_meta, device=device, amp_dtype=amp_dtype,
            uni_omr_mod=uni_omr, uni_omr_v2=uni_omr_v2, eval_s31=eval_s31,
        )
        _atomic_write_json(
            run_file,
            {"exp_id": EXP_ID, "status": "RUNNING", "completed_arms": list(results.keys()), "updated_at": _utc_now()},
        )
        _upload_json(hf_token, run_file, HF_RUN_PATH)

    # Compute gate summary deltas
    a0_s31 = results["arm0_bekern_v2a"]["ema_eval"]["suite31_hw"]
    a1_s31 = results["arm1_bekern3_core"]["ema_eval"]["suite31_hw"]
    a0_mtp = results["arm0_bekern_v2a"]["ema_eval"]["mtp4_throughput"]
    a1_mtp = results["arm1_bekern3_core"]["ema_eval"]["mtp4_throughput"]

    def _delta(k: str) -> Optional[float]:
        v0 = a0_s31.get(k, {}).get("strict_note_f1")
        v1 = a1_s31.get(k, {}).get("strict_note_f1")
        return round(v1 - v0, 4) if (v0 is not None and v1 is not None) else None

    nps0 = float(a0_mtp.get("effective_notes_per_sec") or 0.0)
    nps1 = float(a1_mtp.get("effective_notes_per_sec") or 0.0)
    mtp_gain_pct = round(100.0 * (nps1 / max(1e-6, nps0) - 1.0), 2) if nps0 > 0 else None

    eval_payload = {
        "exp_id": EXP_ID,
        "wave": 13,
        "started_at": started_at,
        "completed_at": _utc_now(),
        "device": str(device),
        "amp_dtype": str(amp_dtype),
        "steps_per_arm": STEPS_PER_ARM,
        "phase0_audit": audit,
        "deltas_arm1_minus_arm0": {
            "pr_clean": _delta("pr_clean"),
            "pr_photo": _delta("pr_photo"),
            "hw_clean": _delta("hw_clean"),
            "hw_photo": _delta("hw_photo"),
            "all45_clean": _delta("all45_clean"),
            "all45_photo": _delta("all45_photo"),
            "cclef_key_heavy": _delta("cclef_key_heavy"),
            "mtp4_effective_notes_per_sec_gain_pct": mtp_gain_pct,
        },
        "arms": results,
    }
    eval_file = out_dir / f"{EXP_ID}.eval.json"
    _atomic_write_json(eval_file, eval_payload)
    _atomic_write_json(run_file, {"exp_id": EXP_ID, "status": "COMPLETED", "completed_at": eval_payload["completed_at"]})
    _upload_json(hf_token, eval_file, HF_EVAL_PATH)
    _upload_json(hf_token, run_file, HF_RUN_PATH)


if __name__ == "__main__":
    main()
