"""Deterministic, lossless canonicalizer for `bekern` (`v1` and `v2a`) token streams.

Closes Gap 1 (I1 Canonical `bekern` Label-Normalization): eliminates duplicate
surface forms for identical musical notation while preserving exact decoded
note multisets and MusicXML round-trip semantics (0 Strict Note F1 loss).

Canonicalization passes (applied per `<sys>` system):
  1. Whitespace / empty-token cleanup.
  2. Leading staff-header canonicalization: enforces `(clef, key, time)` order
     per staff block and removes redundant beat-0 `time-*` / duplicate `key*` /
     duplicate `clef-*` echoes that `decode_bekern` ignores or overwrites with
     an identical value.
  3. Per-measure, per-`<staff>` voice regrouping (I1): regroups interleaved
     `voice-k` segments into strictly ascending `voice-1`, `voice-2`, ... order,
     preserving measure-scoped attributes attached to each voice run.
  4. Mid-measure header-run ordering (`clef -> key -> time`) and deduplication
     of consecutive identical direction/attribute tokens (`dyn-*`, `hairpin-*`,
     `ottava-*`, `volta-*`, `bar-repeat-left`, `clef-*`, `key*`, `time-*`).
  5. Note/rest modifier-tail canonicalization: enforces canonical order matching
     `mscz-rs` (`alter -> dot* -> slur-start -> slur-stop -> stacc -> accent ->
     tenuto -> fermata -> art-* (fixed order) -> cue -> xstaff-* -> tie-start ->
     tie-stop`) and deduplicates repeated non-`dot` modifiers on the same event.
  6. Chord pitch ordering: sorts same-onset chord runs strictly ascending by
     `(midi_pitch, diatonic_pitch, alter, duration_rank, tail)`, keeping `grace`
     on the root tone and `chord` before every non-root tone.
"""

from __future__ import annotations

from typing import Optional

DURATIONS = frozenset(
    {
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
)

DURATION_ORDER: dict[str, int] = {
    "breve": 0,
    "whole": 1,
    "half": 2,
    "quarter": 3,
    "eighth": 4,
    "16th": 5,
    "32nd": 6,
    "64th": 7,
    "128th": 8,
}

STEPS: dict[str, int] = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}

STEP_DIATONIC: dict[str, int] = {
    "C": 0,
    "D": 1,
    "E": 2,
    "F": 3,
    "G": 4,
    "A": 5,
    "B": 6,
}

ALTERS: dict[str, int] = {
    "a-2": -2,
    "a-1": -1,
    "a0": 0,
    "a1": 1,
    "a2": 2,
}

CLOSING_BARLINES = frozenset(
    {
        "bar",
        "bar-double",
        "bar-light-heavy",
        "bar-repeat-right",
    }
)

# Canonical modifier order matching `mscz-rs/src/musicxml_import.rs` + `bekern.rs`.
_CORE_ARTS_ORDER: dict[str, int] = {
    "stacc": 0,
    "accent": 1,
    "tenuto": 2,
    "fermata": 3,
}

_V2_ARTS_ORDER: dict[str, int] = {
    "art-staccatissimo": 0,
    "art-soft-accent": 1,
    "art-trill": 2,
    "art-tremolo": 3,
    "art-mordent": 4,
    "art-turn": 5,
    "art-up-bow": 6,
    "art-down-bow": 7,
    "art-arpeggiate": 8,
}

_SLUR_MODS = frozenset(
    {
        "slur-start",
        "slur-stop",
        "slur-1-open",
        "slur-1-close",
        "slur-2-open",
        "slur-2-close",
        "slur-3-open",
        "slur-3-close",
        "slur-4-open",
        "slur-4-close",
    }
)

_TIES = frozenset({"tie-start", "tie-stop"})


def _is_octave(tok: str) -> bool:
    if not tok:
        return False
    if tok[0] == "-":
        return len(tok) > 1 and tok[1:].isdigit()
    return tok.isdigit()


def _is_header_token(tok: str) -> bool:
    return tok.startswith("clef-") or tok.startswith("key") or tok.startswith("time-")


def _is_voice_token(tok: str) -> Optional[int]:
    if not tok.startswith("voice-"):
        return None
    suffix = tok[6:]
    if not suffix.isdigit():
        return None
    val = int(suffix)
    return val if val >= 1 else None


def _is_measure_attr_token(tok: str) -> bool:
    return (
        tok == "bar-repeat-left"
        or tok.startswith("clef-")
        or tok.startswith("key")
        or tok.startswith("time-")
        or tok.startswith("dyn-")
        or tok.startswith("hairpin-")
        or tok.startswith("ottava-")
        or tok.startswith("volta-")
    )


def _is_dedupable_consec_token(tok: str) -> bool:
    return (
        tok == "bar-repeat-left"
        or tok.startswith("clef-")
        or tok.startswith("key")
        or tok.startswith("time-")
        or tok.startswith("dyn-")
        or tok.startswith("hairpin-")
        or tok.startswith("ottava-")
        or tok.startswith("volta-")
    )


def _is_tail_token(tok: str, allow_ties: bool = True) -> bool:
    if tok == "dot" or tok in ALTERS:
        return True
    if tok in _SLUR_MODS or tok in _CORE_ARTS_ORDER or tok in _V2_ARTS_ORDER:
        return True
    if tok.startswith("art-") or tok == "cue" or tok.startswith("xstaff-"):
        return True
    if allow_ties and tok in _TIES:
        return True
    return False


def _tail_sort_key(tok: str) -> tuple[int, int, str]:
    if tok in ALTERS:
        return (0, 0, tok)
    if tok == "dot":
        return (1, 0, tok)
    if tok == "slur-start":
        return (2, 0, tok)
    if tok == "slur-stop":
        return (2, 1, tok)
    if tok.startswith("slur-"):
        is_close = 1 if tok.endswith("-close") else 0
        return (2, 2 + is_close, tok)
    if tok in _CORE_ARTS_ORDER:
        return (3, _CORE_ARTS_ORDER[tok], tok)
    if tok in _V2_ARTS_ORDER:
        return (4, _V2_ARTS_ORDER[tok], tok)
    if tok.startswith("art-"):
        return (4, 100, tok)
    if tok == "cue":
        return (5, 0, tok)
    if tok == "xstaff-up":
        return (6, 0, tok)
    if tok == "xstaff-down":
        return (6, 1, tok)
    if tok.startswith("xstaff-"):
        return (6, 2, tok)
    if tok == "tie-start":
        return (7, 0, tok)
    if tok == "tie-stop":
        return (7, 1, tok)
    return (8, 0, tok)


def _header_rank(tok: str) -> int:
    if tok.startswith("clef-"):
        return 0
    if tok.startswith("key"):
        return 1
    if tok.startswith("time-"):
        return 2
    return 3


def _canonicalize_leading_headers(header_tokens: list[str]) -> list[str]:
    """Normalize leading staff headers into canonical `(clef, key, time)` blocks.

    In `mscz-rs/src/decode.rs`, the leading header loop:
      - creates one staff entry per `clef-*`
      - sets `leading_keys.last_mut()` on each `key*`
      - sets `time` on the FIRST `time-*` and ignores every subsequent `time-*`.

    Therefore:
      - Within each staff block (from one `clef-*` up to the next `clef-*`),
        reorder `(clef, key, time)` canonically.
      - Drop redundant duplicate `key*` tokens within the same staff block when
        they repeat the active key (e.g., beat-0 `key-2` echo after staff 2's
        `key-2`). If a second distinct `key*` appears (rare), keep only the
        effective last value (which is what `decode.rs` keeps).
      - Keep at most one `time-*` per staff block (the first one, matching
        `decode.rs` / `to_bekern` staff-triple structure); drop redundant extra
        beat-0 `time-*` echoes appended after the last staff's `time-*`.
    """
    if not header_tokens:
        return []

    # Partition into optional pre-clef tokens + one block per `clef-*`.
    pre_clef: list[str] = []
    blocks: list[list[str]] = []
    for tok in header_tokens:
        if tok.startswith("clef-"):
            blocks.append([tok])
        elif blocks:
            blocks[-1].append(tok)
        else:
            pre_clef.append(tok)

    out: list[str] = []
    if pre_clef:
        keys = [t for t in pre_clef if t.startswith("key")]
        times = [t for t in pre_clef if t.startswith("time-")]
        if keys:
            out.append(keys[-1])
        if times:
            out.append(times[0])

    for blk in blocks:
        clef_tok = blk[0]
        keys = [t for t in blk[1:] if t.startswith("key")]
        times = [t for t in blk[1:] if t.startswith("time-")]
        out.append(clef_tok)
        if keys:
            # `decode.rs` overwrites `leading_keys.last_mut()`, so last wins.
            out.append(keys[-1])
        if times:
            # `to_bekern` emits one `time-*` per staff; any second `time-*`
            # in the same block is a beat-0 echo ignored by `decode.rs`.
            out.append(times[0])

    return out


def _canonicalize_mid_headers_and_dups(tokens: list[str]) -> list[str]:
    """Order contiguous mid-measure `(clef, key, time)` runs and dedup consec dirs."""
    n = len(tokens)
    out: list[str] = []
    i = 0
    while i < n:
        if _is_header_token(tokens[i]):
            j = i
            while j < n and _is_header_token(tokens[j]):
                j += 1
            run = tokens[i:j]
            if len(run) > 1:
                # Stable sort by (clef=0, key=1, time=2)
                run = sorted(run, key=_header_rank)
            for t in run:
                if out and out[-1] == t and _is_dedupable_consec_token(t):
                    continue
                out.append(t)
            i = j
        else:
            t = tokens[i]
            if out and out[-1] == t and _is_dedupable_consec_token(t):
                i += 1
                continue
            out.append(t)
            i += 1
    return out


def _canonicalize_voice_order_segment(seg_tokens: list[str]) -> list[str]:
    """Regroup interleaved `voice-k` runs in a single measure/staff segment (I1)."""
    prev_v = 0
    is_mono = True
    for tok in seg_tokens:
        v = _is_voice_token(tok)
        if v is not None:
            if v <= prev_v:
                is_mono = False
                break
            prev_v = v
    if is_mono:
        return seg_tokens

    leading_orphans: list[str] = []
    voice_first_attrs: dict[int, list[str]] = {}
    voice_events: dict[int, list[str]] = {}

    cur_voice: Optional[int] = None
    cur_buf: list[str] = []

    def peel_trailing_attrs(buf: list[str]) -> tuple[list[str], list[str]]:
        cut = len(buf)
        while cut > 0 and _is_measure_attr_token(buf[cut - 1]):
            cut -= 1
        return buf[:cut], buf[cut:]

    for tok in seg_tokens:
        v = _is_voice_token(tok)
        if v is not None:
            head, attrs = peel_trailing_attrs(cur_buf)
            if cur_voice is None:
                leading_orphans.extend(head)
            else:
                voice_events[cur_voice].extend(head)

            if v not in voice_events:
                voice_first_attrs[v] = list(attrs)
                voice_events[v] = []
            else:
                voice_events[v].extend(attrs)
            cur_voice = v
            cur_buf = []
        else:
            cur_buf.append(tok)

    if cur_voice is None:
        leading_orphans.extend(cur_buf)
    else:
        voice_events[cur_voice].extend(cur_buf)

    out: list[str] = list(leading_orphans)
    for v in sorted(voice_events.keys()):
        out.extend(voice_first_attrs[v])
        out.append(f"voice-{v}")
        out.extend(voice_events[v])
    return out


def _canonicalize_tail(tail: list[str]) -> list[str]:
    """Sort and deduplicate non-`dot` modifiers in a note/rest tail."""
    if not tail:
        return []
    sorted_tail = sorted(tail, key=_tail_sort_key)
    out: list[str] = []
    seen_non_dot: set[str] = set()
    for t in sorted_tail:
        if t == "dot":
            out.append(t)
        elif t not in seen_non_dot:
            seen_non_dot.add(t)
            out.append(t)
    return out


def _canonicalize_notes_and_chords(seg_tokens: list[str]) -> list[str]:
    """Canonicalize note/rest modifier tails and sort chord runs ascending by pitch."""
    n = len(seg_tokens)
    out: list[str] = []
    i = 0

    while i < n:
        tok = seg_tokens[i]

        # Rest event: `rest <dur> [tail...]`
        if tok == "rest" and i + 1 < n and seg_tokens[i + 1] in DURATIONS:
            dur = seg_tokens[i + 1]
            i += 2
            tail: list[str] = []
            while i < n and _is_tail_token(seg_tokens[i], allow_ties=False):
                tail.append(seg_tokens[i])
                i += 1
            out.append("rest")
            out.append(dur)
            out.extend(_canonicalize_tail(tail))
            continue

        # Potential pitched note or chord run (optionally preceded by `grace` / `grace-slash`)
        grace_tok: Optional[str] = None
        start_i = i
        if tok in ("grace", "grace-slash") and i + 3 < n and seg_tokens[i + 1] in DURATIONS and seg_tokens[i + 2] in STEPS and _is_octave(seg_tokens[i + 3]):
            grace_tok = tok
            i += 1
            tok = seg_tokens[i]

        if (
            tok in DURATIONS
            and i + 2 < n
            and seg_tokens[i + 1] in STEPS
            and _is_octave(seg_tokens[i + 2])
        ):
            # Parse root tone + any immediately following `chord` / `<chord>` tones
            chord_tones: list[tuple[tuple[int, int, int, int, tuple[str, ...]], list[str]]] = []
            chord_marker = "chord"

            while True:
                dur = seg_tokens[i]
                step = seg_tokens[i + 1]
                oct_s = seg_tokens[i + 2]
                i += 3
                tail_raw: list[str] = []
                while i < n and _is_tail_token(seg_tokens[i], allow_ties=True):
                    tail_raw.append(seg_tokens[i])
                    i += 1
                canon_tail = _canonicalize_tail(tail_raw)

                # Extract alter (if any; placed first by `_canonicalize_tail`)
                alt_val = 0
                if canon_tail and canon_tail[0] in ALTERS:
                    alt_val = ALTERS[canon_tail[0]]

                oct_i = int(oct_s)
                midi = (oct_i + 1) * 12 + STEPS[step] + alt_val
                dia = oct_i * 7 + STEP_DIATONIC[step]
                sort_key = (
                    midi,
                    dia,
                    alt_val,
                    DURATION_ORDER.get(dur, 99),
                    tuple(canon_tail),
                )
                tone_tokens = [dur, step, oct_s, *canon_tail]
                chord_tones.append((sort_key, tone_tokens))

                # Check if next event is a chord continuation
                if (
                    i + 3 < n
                    and seg_tokens[i] in ("chord", "<chord>")
                    and seg_tokens[i + 1] in DURATIONS
                    and seg_tokens[i + 2] in STEPS
                    and _is_octave(seg_tokens[i + 3])
                ):
                    chord_marker = seg_tokens[i]
                    i += 1
                else:
                    break

            if len(chord_tones) > 1:
                chord_tones.sort(key=lambda item: item[0])

            if grace_tok is not None:
                out.append(grace_tok)
            for idx_t, (_, tone_toks) in enumerate(chord_tones):
                if idx_t > 0:
                    out.append(chord_marker)
                out.extend(tone_toks)
            continue

        # Any other token (revert if grace_tok wasn't consumed)
        i = start_i
        out.append(seg_tokens[i])
        i += 1

    return out


def _canonicalize_single_system(tokens: list[str]) -> list[str]:
    """Canonicalize a single system (no `<sys>` separators)."""
    cleaned = [t.strip() for t in tokens if t and t.strip()]
    if not cleaned:
        return []

    # 1. Split leading headers (`clef-*`, `key*`, `time-*`)
    i = 0
    n = len(cleaned)
    while i < n and _is_header_token(cleaned[i]):
        i += 1
    headers = _canonicalize_leading_headers(cleaned[:i])
    body = cleaned[i:]

    # 2. Split body into measures delimited by closing barlines
    out: list[str] = list(headers)
    cur_m: list[str] = []

    def flush_measure(m_toks: list[str], barline: Optional[str]) -> None:
        # Split measure by `<staff>` so v2a multi-staff voice numbering is
        # canonicalized independently per staff.
        staff_segs: list[list[str]] = []
        sc: list[str] = []
        for t in m_toks:
            if t == "<staff>":
                staff_segs.append(sc)
                sc = []
            else:
                sc.append(t)
        staff_segs.append(sc)

        for s_idx, seg in enumerate(staff_segs):
            if s_idx > 0:
                out.append("<staff>")
            seg = _canonicalize_voice_order_segment(seg)
            seg = _canonicalize_mid_headers_and_dups(seg)
            seg = _canonicalize_notes_and_chords(seg)
            out.extend(seg)
        if barline is not None:
            out.append(barline)

    for tok in body:
        if tok in CLOSING_BARLINES:
            flush_measure(cur_m, tok)
            cur_m = []
        else:
            cur_m.append(tok)
    if cur_m:
        flush_measure(cur_m, None)

    return out


def canonicalize_bekern_tokens(tokens: list[str]) -> list[str]:
    """Return the deterministic canonical form of a `bekern` (`v1`/`v2a`) token list.

    Supports both single-system streams and `<sys>`-joined full-page streams.
    Idempotent: `canonicalize_bekern_tokens(canonicalize_bekern_tokens(x)) == canonicalize_bekern_tokens(x)`.
    """
    if not tokens:
        return []

    # Split on `<sys>` if present, canonicalize each system independently.
    if "<sys>" not in tokens:
        return _canonicalize_single_system(tokens)

    systems: list[list[str]] = []
    cur: list[str] = []
    for tok in tokens:
        t = tok.strip()
        if not t:
            continue
        if t == "<sys>":
            systems.append(cur)
            cur = []
        else:
            cur.append(t)
    systems.append(cur)

    out: list[str] = []
    for s_idx, sys_toks in enumerate(systems):
        if s_idx > 0:
            out.append("<sys>")
        out.extend(_canonicalize_single_system(sys_toks))
    return out


def canonicalize_bekern_text(text: str) -> str:
    """Canonicalize a whitespace-delimited `bekern` text string."""
    toks = text.split()
    canon = canonicalize_bekern_tokens(toks)
    return " ".join(canon)


def diff_canonicalization(tokens: list[str]) -> dict[str, int]:
    """Return per-category counts of canonicalization changes applied to `tokens`."""
    counts = {
        "changed": 0,
        "leading_header": 0,
        "voice_order": 0,
        "mid_header_or_dup": 0,
        "note_or_chord": 0,
        "token_delta": 0,
    }
    if not tokens:
        return counts

    systems: list[list[str]] = []
    cur: list[str] = []
    for tok in tokens:
        t = tok.strip()
        if not t:
            continue
        if t == "<sys>":
            systems.append(cur)
            cur = []
        else:
            cur.append(t)
    systems.append(cur)

    total_after = 0
    for s_idx, sys_toks in enumerate(systems):
        if s_idx > 0:
            total_after += 1
        i = 0
        n = len(sys_toks)
        while i < n and _is_header_token(sys_toks[i]):
            i += 1
        hdr_orig = sys_toks[:i]
        hdr_canon = _canonicalize_leading_headers(hdr_orig)
        if hdr_canon != hdr_orig:
            counts["leading_header"] += 1
        total_after += len(hdr_canon)

        body = sys_toks[i:]
        cur_m: list[str] = []

        def check_measure(m_toks: list[str], barline: Optional[str]) -> None:
            nonlocal total_after
            staff_segs: list[list[str]] = []
            sc: list[str] = []
            for t in m_toks:
                if t == "<staff>":
                    staff_segs.append(sc)
                    sc = []
                else:
                    sc.append(t)
            staff_segs.append(sc)
            for si, seg in enumerate(staff_segs):
                if si > 0:
                    total_after += 1
                s1 = _canonicalize_voice_order_segment(seg)
                if s1 != seg:
                    counts["voice_order"] += 1
                s2 = _canonicalize_mid_headers_and_dups(s1)
                if s2 != s1:
                    counts["mid_header_or_dup"] += 1
                s3 = _canonicalize_notes_and_chords(s2)
                if s3 != s2:
                    counts["note_or_chord"] += 1
                total_after += len(s3)
            if barline is not None:
                total_after += 1

        for tok in body:
            if tok in CLOSING_BARLINES:
                check_measure(cur_m, tok)
                cur_m = []
            else:
                cur_m.append(tok)
        if cur_m:
            check_measure(cur_m, None)

    canon_full = canonicalize_bekern_tokens(tokens)
    if canon_full != tokens:
        counts["changed"] = 1
    counts["token_delta"] = abs(len(canon_full) - len(tokens))
    return counts
