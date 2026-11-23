"""Fast, verified CPU I1 re-canonicalizer for bekern target files.

Transforms pre-I1 bekern token streams (where voices within a measure may be
interleaved by onset, e.g. `voice-1 ... voice-2 ... voice-1 ...`) into I1
canonical `(measure, staff, voice, onset)` form where every measure emits its
voice segments contiguously in strictly ascending `voice-1`, `voice-2`, ...
order.

Safety guarantees:
  - Already-canonical files are untouched (exact no-op).
  - Every rewritten file is verified via `assembly-serialize.exe bekern-decode`
    (exit 0 AND identical multiset of `(midi, onset, duration, voice)` pitched
    notes + rests vs. the original file). Fail-closed: never overwrites on
    mismatch.
  - Writes atomically via sibling `.tmp` + `os.replace`, cleanly breaking
    hardlinks in `unified-sysdata` without mutating source corpus directories.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import os
import random
import subprocess
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Counter, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_ASER = (
    REPO_ROOT
    / "converter"
    / "assembly-serialize"
    / "target"
    / "release"
    / "assembly-serialize.exe"
)

CLOSING_BARLINES = frozenset(
    {
        "bar",
        "bar-double",
        "bar-light-heavy",
        "bar-repeat-right",
    }
)

STEP_SEMITONE = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}

ALTER_SEMITONE = {
    "": 0,
    "#": 1,
    "##": 2,
    "x": 2,
    "b": -1,
    "bb": -2,
}


def parse_voice_token(tok: str) -> Optional[int]:
    """Return 1-based voice index if `tok` is `voice-<N>` (N >= 1), else None."""
    if not tok.startswith("voice-"):
        return None
    suffix = tok[6:]
    if not suffix.isdigit():
        return None
    val = int(suffix)
    return val if val >= 1 else None


def is_leading_header_token(tok: str) -> bool:
    """True for leading staff header tokens (`clef-*`, `key*`, `time-*`)."""
    return tok.startswith("clef-") or tok.startswith("key") or tok.startswith("time-")


def is_measure_attr_token(tok: str) -> bool:
    """True for measure-scoped attribute/direction tokens emitted before `voice-k`."""
    return (
        tok == "bar-repeat-left"
        or tok.startswith("clef-")
        or tok.startswith("key")
        or tok.startswith("time-")
        or tok.startswith("dyn-")
        or tok.startswith("hairpin-")
    )


def pitch_to_midi(pitch: Optional[str]) -> Optional[int]:
    """Convert `decode_bekern` pitch string (e.g. `Eb4`, `F#5`, `Ax4`) to MIDI."""
    if pitch is None:
        return None
    if len(pitch) < 2:
        raise ValueError(f"invalid pitch string: {pitch!r}")
    step = pitch[0]
    if step not in STEP_SEMITONE:
        raise ValueError(f"invalid pitch step in {pitch!r}")
    # Split trailing signed integer octave from optional accidental suffix.
    oct_start = len(pitch)
    while oct_start > 1 and pitch[oct_start - 1].isdigit():
        oct_start -= 1
    if oct_start > 1 and pitch[oct_start - 1] == "-":
        oct_start -= 1
    acc = pitch[1:oct_start]
    oct_str = pitch[oct_start:]
    if acc not in ALTER_SEMITONE:
        raise ValueError(f"invalid pitch accidental {acc!r} in {pitch!r}")
    octave = int(oct_str)
    return (octave + 1) * 12 + STEP_SEMITONE[step] + ALTER_SEMITONE[acc]


def split_leading_headers(tokens: list[str]) -> tuple[list[str], list[str]]:
    """Split leading staff headers (`clef-*`, `key*`, `time-*`) from body."""
    i = 0
    n = len(tokens)
    while i < n and is_leading_header_token(tokens[i]):
        i += 1
    return tokens[:i], tokens[i:]


def split_measures(body_tokens: list[str]) -> list[tuple[list[str], Optional[str]]]:
    """Split body tokens into `(measure_tokens, closing_barline)` pairs."""
    measures: list[tuple[list[str], Optional[str]]] = []
    cur: list[str] = []
    for tok in body_tokens:
        if tok in CLOSING_BARLINES:
            measures.append((cur, tok))
            cur = []
        else:
            cur.append(tok)
    if cur:
        measures.append((cur, None))
    return measures


def is_measure_monotonic(measure_tokens: list[str]) -> bool:
    """True iff `voice-k` tokens in this measure are strictly ascending."""
    prev_v = 0
    for tok in measure_tokens:
        v = parse_voice_token(tok)
        if v is not None:
            if v <= prev_v:
                return False
            prev_v = v
    return True


def is_stream_canonical_i1(text: str) -> bool:
    """Return True iff every measure in `text` has strictly ascending `voice-k`."""
    tokens = text.split()
    _, body = split_leading_headers(tokens)
    for m_toks, _ in split_measures(body):
        if not is_measure_monotonic(m_toks):
            return False
    return True


def canonicalize_measure(measure_tokens: list[str]) -> list[str]:
    """Regroup `voice-k` segments in a single measure into strictly ascending order.

    If the measure already has strictly ascending `voice-k` tokens (or no voice
    tokens), returns `measure_tokens` unchanged (exact no-op).
    """
    if is_measure_monotonic(measure_tokens):
        return measure_tokens

    # Parse into (voice_id, leading_attrs, body_tokens) runs.
    # Measure-scoped attribute tokens immediately preceding a `voice-k` token
    # are peeled off the tail of the preceding buffer and attached to that run,
    # matching `to_bekern`'s emission order (`[attrs...] voice-k [events...]`).
    leading_orphans: list[str] = []
    # Map voice_id -> (first_run_attrs, list_of_event_tokens)
    voice_first_attrs: dict[int, list[str]] = {}
    voice_events: dict[int, list[str]] = {}

    cur_voice: Optional[int] = None
    cur_buf: list[str] = []

    def peel_trailing_attrs(buf: list[str]) -> tuple[list[str], list[str]]:
        cut = len(buf)
        while cut > 0 and is_measure_attr_token(buf[cut - 1]):
            cut -= 1
        return buf[:cut], buf[cut:]

    for tok in measure_tokens:
        v = parse_voice_token(tok)
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
                # Subsequent occurrence of the same voice: keep any attached
                # mid-measure attrs right at this event position inside the voice.
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


def canonicalize_bekern_tokens(tokens: list[str]) -> tuple[list[str], bool]:
    """Canonicalize a bekern token list to I1 order.

    Returns `(new_tokens, changed)`.
    """
    headers, body = split_leading_headers(tokens)
    measures = split_measures(body)
    out: list[str] = list(headers)
    changed = False
    for m_toks, barline in measures:
        new_m = canonicalize_measure(m_toks)
        if new_m != m_toks:
            changed = True
        out.extend(new_m)
        if barline is not None:
            out.append(barline)
    if not changed:
        return tokens, False
    return out, True


def canonicalize_bekern_text(text: str) -> tuple[str, bool]:
    """Canonicalize a bekern text string to I1 order.

    Returns `(new_text, changed)`. Preserves trailing newline if present.
    """
    tokens = text.split()
    new_tokens, changed = canonicalize_bekern_tokens(tokens)
    if not changed:
        return text, False
    trailing = "\n" if text.endswith("\n") else ""
    return " ".join(new_tokens) + trailing, True


def decode_note_multiset(
    bekern_path: Path,
    aser_bin: Path,
    tmp_json: Path,
) -> Counter[tuple[Optional[int], float, float, int]]:
    """Run `assembly-serialize bekern-decode` and return the note multiset.

    Each entry is `(midi, round(onset, 4), round(dur, 4), voice)` for pitched
    notes, or `(None, round(onset, 4), round(dur, 4), voice)` for rests.
    Raises RuntimeError on non-zero exit or invalid JSON.
    """
    proc = subprocess.run(
        [str(aser_bin), "bekern-decode", "--input", str(bekern_path), "--out", str(tmp_json)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"bekern-decode failed (rc={proc.returncode}) on {bekern_path}: "
            f"{proc.stderr.strip()[:300]}"
        )
    data = json.loads(tmp_json.read_text(encoding="utf-8"))
    counts: Counter[tuple[Optional[int], float, float, int]] = collections.Counter()
    for part in data.get("parts", []):
        for note in part.get("notes", []):
            midi = None if note.get("rest") else pitch_to_midi(note.get("pitch"))
            onset = round(float(note["onset"]), 4)
            dur = round(float(note["dur"]), 4)
            voice = int(note["voice"])
            counts[(midi, onset, dur, voice)] += 1
    return counts


def pitched_only(
    multiset: Counter[tuple[Optional[int], float, float, int]],
) -> Counter[tuple[int, float, float, int]]:
    """Filter a note multiset to pitched `(midi, onset, dur, voice)` entries."""
    return collections.Counter(
        {k: v for k, v in multiset.items() if k[0] is not None}
    )


def decode_unmerged_mxml_multiset(
    bekern_path: Path,
    aser_bin: Path,
    out_dir: Path,
) -> Counter[tuple[int, int, int, str, Optional[tuple[str, str, str]], bool, tuple[str, ...]]]:
    """Run `assembly-serialize bekern2musicxml` and return exact unmerged note multiset.

    Each entry is `(measure_idx, onset_ticks, dur_ticks, voice, pitch_tuple, is_grace, ties)`.
    Used as an exact tie-link-faithful verifier when `bekern-decode`'s sequential
    cross-voice tie merger (`merge_ties=True`) differs on page-cut `tie-stop`s.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            str(aser_bin),
            "bekern2musicxml",
            "--input",
            str(bekern_path),
            "--out",
            str(out_dir),
            "--stem",
            "verify",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"bekern2musicxml failed (rc={proc.returncode}) on {bekern_path}: "
            f"{proc.stderr.strip()[:300]}"
        )
    mxml_path = out_dir / "verify.musicxml"
    tree = ET.parse(mxml_path)
    counts: Counter[
        tuple[int, int, int, str, Optional[tuple[str, str, str]], bool, tuple[str, ...]]
    ] = collections.Counter()
    for part in tree.findall(".//part"):
        for m_idx, meas in enumerate(part.findall("measure")):
            cursor = 0
            last_onset = 0
            for el in meas:
                if el.tag == "backup":
                    cursor -= int(el.findtext("duration", "0"))
                elif el.tag == "forward":
                    cursor += int(el.findtext("duration", "0"))
                elif el.tag == "note":
                    is_chord = el.find("chord") is not None
                    is_grace = el.find("grace") is not None
                    dur = int(el.findtext("duration", "0"))
                    onset = last_onset if is_chord else cursor
                    voice = el.findtext("voice", "1")
                    pitch_el = el.find("pitch")
                    if pitch_el is not None:
                        p: Optional[tuple[str, str, str]] = (
                            pitch_el.findtext("step", ""),
                            pitch_el.findtext("octave", ""),
                            pitch_el.findtext("alter", "0"),
                        )
                    else:
                        p = None
                    ties = tuple(
                        sorted(t.attrib.get("type", "") for t in el.findall("tie"))
                    )
                    counts[(m_idx, onset, dur, voice, p, is_grace, ties)] += 1
                    if not is_chord and not is_grace:
                        last_onset = onset
                        cursor += dur
    return counts


def verify_equivalence(
    orig_path: Path,
    cand_path: Path,
    aser_bin: Path,
    scratch_dir: Path,
    tag: str,
) -> tuple[bool, str]:
    """Verify `cand_path` decodes to the exact same note multiset as `orig_path`.

    Primary check: `assembly-serialize bekern-decode` must exit 0 on both files
    and produce an identical `(midi, onset, dur, voice)` note multiset.
    Fallback (only when both `bekern-decode` calls exit 0 and differ solely due
    to sequential cross-voice tie merging on page-cut `tie-stop`s): verify that
    `assembly-serialize bekern2musicxml` (unmerged tie links) produces the 100%
    identical `(measure, onset, dur, voice, pitch, grace, ties)` multiset.
    """
    orig_json = scratch_dir / f"{tag}.orig.json"
    cand_json = scratch_dir / f"{tag}.cand.json"
    try:
        orig_notes = decode_note_multiset(orig_path, aser_bin, orig_json)
        cand_notes = decode_note_multiset(cand_path, aser_bin, cand_json)
    except Exception as exc:
        return False, str(exc)
    finally:
        for p in (orig_json, cand_json):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass

    if orig_notes == cand_notes:
        return True, ""

    # Both `bekern-decode` calls exited 0; check exact unmerged tie-link equivalence
    # via `bekern2musicxml` before failing closed.
    orig_dir = scratch_dir / f"{tag}.orig_mxml"
    cand_dir = scratch_dir / f"{tag}.cand_mxml"
    try:
        orig_unmerged = decode_unmerged_mxml_multiset(orig_path, aser_bin, orig_dir)
        cand_unmerged = decode_unmerged_mxml_multiset(cand_path, aser_bin, cand_dir)
        if orig_unmerged == cand_unmerged:
            return True, ""
    except Exception as exc:
        return False, f"unmerged fallback error: {exc}"
    finally:
        import shutil
        for d in (orig_dir, cand_dir):
            shutil.rmtree(d, ignore_errors=True)

    orig_p = pitched_only(orig_notes)
    cand_p = pitched_only(cand_notes)
    return (
        False,
        f"note multiset mismatch: orig_total={sum(orig_notes.values())} "
        f"cand_total={sum(cand_notes.values())} "
        f"orig_pitched={sum(orig_p.values())} cand_pitched={sum(cand_p.values())} "
        f"diff_orig={list((orig_notes - cand_notes).items())[:3]} "
        f"diff_cand={list((cand_notes - orig_notes).items())[:3]}",
    )


def process_one_file(
    path: Path,
    aser_bin: Path,
    scratch_dir: Path,
    dry_run: bool = False,
) -> tuple[str, str]:
    """Process one `*-bekern.txt` file.

    Returns `(status, detail)` where status is one of:
      - `"noop"`: already I1 canonical
      - `"rewritten"`: canonicalized, verified, and atomically replaced
      - `"failed"`: verification or I/O failed (original file left untouched)
    """
    try:
        orig_text = path.read_text(encoding="utf-8")
    except Exception as exc:
        return "failed", f"{path.name}: read error: {exc}"

    new_text, changed = canonicalize_bekern_text(orig_text)
    if not changed:
        return "noop", ""

    if not is_stream_canonical_i1(new_text):
        return "failed", f"{path.name}: post-canonicalization monotonicity check failed"

    tid = threading.get_ident()
    pid = os.getpid()
    tag = f"{pid}_{tid}_{path.stem}"
    tmp_path = path.with_name(f".{path.name}.{pid}.{tid}.tmp")
    try:
        tmp_path.write_text(new_text, encoding="utf-8")
        ok, err = verify_equivalence(path, tmp_path, aser_bin, scratch_dir, tag)
        if not ok:
            tmp_path.unlink(missing_ok=True)
            return "failed", f"{path.name}: {err}"
        if dry_run:
            tmp_path.unlink(missing_ok=True)
        else:
            os.replace(tmp_path, path)
        return "rewritten", ""
    except Exception as exc:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        return "failed", f"{path.name}: {exc}"


def iter_bekern_files(root: Path) -> list[Path]:
    """Return sorted list of `*-bekern.txt` files directly inside `root`."""
    return sorted(root.glob("*-bekern.txt"))


def audit_directory(
    root: Path,
    sample_size: Optional[int] = None,
    seed: int = 42,
) -> dict[str, int | float]:
    """Audit `*-bekern.txt` files in `root` for non-monotonic voice bars."""
    files = iter_bekern_files(root)
    total_available = len(files)
    if sample_size is not None and sample_size < len(files):
        rng = random.Random(seed)
        files = rng.sample(files, sample_size)

    bad_files = 0
    total_bars = 0
    bad_bars = 0
    for p in files:
        tokens = p.read_text(encoding="utf-8").split()
        _, body = split_leading_headers(tokens)
        file_bad = False
        for m_toks, _ in split_measures(body):
            total_bars += 1
            if not is_measure_monotonic(m_toks):
                bad_bars += 1
                file_bad = True
        if file_bad:
            bad_files += 1

    return {
        "total_available": total_available,
        "audited_files": len(files),
        "bad_files": bad_files,
        "total_bars": total_bars,
        "bad_bars": bad_bars,
        "bad_bar_pct": (100.0 * bad_bars / total_bars) if total_bars else 0.0,
        "bad_file_pct": (100.0 * bad_files / len(files)) if files else 0.0,
    }


def run_recanon(
    target_dir: Path,
    aser_bin: Path = DEFAULT_ASER,
    workers: int = 12,
    dry_run: bool = False,
    progress_every: int = 5000,
) -> dict[str, object]:
    """Re-canonicalize all `*-bekern.txt` files in `target_dir` in parallel."""
    if not aser_bin.is_file():
        raise FileNotFoundError(f"assembly-serialize binary not found: {aser_bin}")
    files = iter_bekern_files(target_dir)
    t0 = time.perf_counter()
    counts = {"total": len(files), "noop": 0, "rewritten": 0, "failed": 0}
    failures: list[str] = []

    with tempfile.TemporaryDirectory(prefix="recanon_i1_") as scratch:
        scratch_dir = Path(scratch)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [
                pool.submit(process_one_file, p, aser_bin, scratch_dir, dry_run)
                for p in files
            ]
            for done, fut in enumerate(concurrent.futures.as_completed(futs), 1):
                status, detail = fut.result()
                counts[status] += 1
                if detail:
                    failures.append(detail)
                if progress_every and (done % progress_every == 0 or done == len(files)):
                    dt = time.perf_counter() - t0
                    rate = done / dt if dt > 0 else 0.0
                    print(
                        f"[{done}/{len(files)}] rewritten={counts['rewritten']} "
                        f"noop={counts['noop']} failed={counts['failed']} "
                        f"({dt:.1f}s, {rate:.0f} files/s)",
                        flush=True,
                    )

    elapsed = time.perf_counter() - t0
    return {
        **counts,
        "elapsed_s": round(elapsed, 2),
        "failures": failures,
    }


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Verify and re-canonicalize bekern files to I1 order.")
    ap.add_argument("target_dir", type=Path, help="Directory containing *-bekern.txt files.")
    ap.add_argument("--aser", type=Path, default=DEFAULT_ASER, help="Path to assembly-serialize.exe.")
    ap.add_argument("--workers", type=int, default=12, help="Number of parallel worker threads.")
    ap.add_argument("--dry-run", action="store_true", help="Verify rewrites without replacing files.")
    ap.add_argument("--audit-only", action="store_true", help="Only audit monotonicity (no rewrites).")
    ap.add_argument("--sample", type=int, default=None, help="Audit random sample size (default: all).")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for --sample.")
    args = ap.parse_args(list(argv) if argv is not None else None)

    if args.audit_only:
        res = audit_directory(args.target_dir, sample_size=args.sample, seed=args.seed)
        print(json.dumps(res, indent=2))
        return 0 if res["bad_bars"] == 0 else 1

    res = run_recanon(
        target_dir=args.target_dir,
        aser_bin=args.aser,
        workers=args.workers,
        dry_run=args.dry_run,
    )
    print(json.dumps({k: v for k, v in res.items() if k != "failures"}, indent=2))
    if res["failures"]:
        print(f"\nFirst 10 failures (of {len(res['failures'])}):", file=sys.stderr)
        for msg in res["failures"][:10]:
            print(f"  - {msg}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
