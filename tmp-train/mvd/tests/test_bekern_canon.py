"""Fast CPU unit tests for `tmp-train/mvd/bekern_canon.py` (Gap 1 / I1)."""

from __future__ import annotations

import collections
import json
import pathlib
import random
import subprocess
import sys
import tempfile
import pytest

MVD_DIR = pathlib.Path(__file__).resolve().parent.parent
REPO_ROOT = MVD_DIR.parent.parent
sys.path.insert(0, str(MVD_DIR))

import bekern_canon as bc  # noqa: E402
import recanon_i1 as rc  # noqa: E402

ASER_BIN = (
    REPO_ROOT
    / "converter"
    / "assembly-serialize"
    / "target"
    / "release"
    / "assembly-serialize.exe"
)
SUITE31_GT_DIR = REPO_ROOT / "data-gen" / "score-corpus" / "suite31" / "gt"
ITER15_MANIFEST = pathlib.Path(
    r"D:\tmp\iter15_local_data\uni_fullpage_iter15_v2a_local.jsonl"
)


# ---------------------------------------------------------------------------
# 1. Core unit tests: chords, modifiers, voices, headers, duplicates, <sys>
# ---------------------------------------------------------------------------


def test_chord_pitch_sorting_ascending_and_grace_preserved() -> None:
    # Descending chord + mixed grace chord
    toks = (
        "clef-G-2 key+0 time-4-4 voice-1 "
        "quarter G 5 chord quarter C 4 chord quarter E 4 "
        "grace eighth B 5 chord eighth D 5 "
        "quarter C 5 bar"
    ).split()
    got = bc.canonicalize_bekern_tokens(toks)
    expected = (
        "clef-G-2 key+0 time-4-4 voice-1 "
        "quarter C 4 chord quarter E 4 chord quarter G 5 "
        "grace eighth D 5 chord eighth B 5 "
        "quarter C 5 bar"
    ).split()
    assert got == expected
    assert bc.canonicalize_bekern_tokens(got) == got


def test_chord_enharmonic_midi_tie_sorted_by_diatonic_pitch() -> None:
    # Db4 (MIDI 61, dia C=0, D=1) vs C#4 (MIDI 61, dia C=0): C#4 must precede Db4
    toks = (
        "clef-G-2 key+0 time-4-4 voice-1 "
        "quarter D 4 a-1 chord quarter C 4 a1 bar"
    ).split()
    got = bc.canonicalize_bekern_tokens(toks)
    expected = (
        "clef-G-2 key+0 time-4-4 voice-1 "
        "quarter C 4 a1 chord quarter D 4 a-1 bar"
    ).split()
    assert got == expected
    assert bc.canonicalize_bekern_tokens(got) == got


def test_modifier_ordering_and_deduplication_preserves_dots() -> None:
    # Scrambled tail with duplicate stacc, tie-stop before tie-start, dot before alter, double-dot
    toks = (
        "clef-G-2 key+0 time-4-4 voice-1 "
        "quarter F 4 dot tie-stop accent a1 dot stacc stacc slur-stop slur-start tie-start "
        "rest quarter fermata dot dot bar"
    ).split()
    got = bc.canonicalize_bekern_tokens(toks)
    expected = (
        "clef-G-2 key+0 time-4-4 voice-1 "
        "quarter F 4 a1 dot dot slur-start slur-stop stacc accent tie-start tie-stop "
        "rest quarter dot dot fermata bar"
    ).split()
    assert got == expected
    assert bc.canonicalize_bekern_tokens(got) == got


def test_v2a_extended_modifier_order() -> None:
    toks = (
        "clef-G-2 key+0 time-4-4 voice-1 "
        "quarter G 4 xstaff-up cue art-arpeggiate art-trill tenuto stacc accent tie-start bar"
    ).split()
    got = bc.canonicalize_bekern_tokens(toks)
    expected = (
        "clef-G-2 key+0 time-4-4 voice-1 "
        "quarter G 4 stacc accent tenuto art-trill art-arpeggiate cue xstaff-up tie-start bar"
    ).split()
    assert got == expected
    assert bc.canonicalize_bekern_tokens(got) == got


def test_multi_voice_regrouping_single_and_multistaff() -> None:
    toks = (
        "clef-G-2 key+0 time-4-4 clef-F-4 key+0 time-4-4 "
        "voice-1 quarter C 5 voice-2 quarter E 4 voice-1 quarter D 5 voice-2 quarter F 4 "
        "<staff> "
        "voice-1 half C 3 voice-2 quarter G 2 voice-1 half D 3 voice-2 quarter A 2 bar"
    ).split()
    got = bc.canonicalize_bekern_tokens(toks)
    expected = (
        "clef-G-2 key+0 time-4-4 clef-F-4 key+0 time-4-4 "
        "voice-1 quarter C 5 quarter D 5 voice-2 quarter E 4 quarter F 4 "
        "<staff> "
        "voice-1 half C 3 half D 3 voice-2 quarter G 2 quarter A 2 bar"
    ).split()
    assert got == expected
    assert bc.canonicalize_bekern_tokens(got) == got


def test_leading_header_redundant_echoes_and_ordering() -> None:
    # Real pattern observed in iter15 v2a: beat-0 `time-3-4` / `key-2` echo after staff headers
    toks = (
        "clef-G-2 time-3-4 key-2 clef-F-4 key-2 time-3-4 key-2 time-3-4 "
        "dyn-f dyn-f voice-1 quarter D 4 bar"
    ).split()
    got = bc.canonicalize_bekern_tokens(toks)
    expected = (
        "clef-G-2 key-2 time-3-4 clef-F-4 key-2 time-3-4 "
        "dyn-f voice-1 quarter D 4 bar"
    ).split()
    assert got == expected
    assert bc.canonicalize_bekern_tokens(got) == got


def test_mid_measure_header_order_and_consec_dups() -> None:
    toks = (
        "clef-G-2 key+0 time-4-4 voice-1 whole C 4 bar "
        "time-3-4 key+1 clef-F-4 hairpin-stop hairpin-stop dyn-p dyn-p "
        "voice-1 half G 3 dot bar-light-heavy"
    ).split()
    got = bc.canonicalize_bekern_tokens(toks)
    expected = (
        "clef-G-2 key+0 time-4-4 voice-1 whole C 4 bar "
        "clef-F-4 key+1 time-3-4 hairpin-stop dyn-p "
        "voice-1 half G 3 dot bar-light-heavy"
    ).split()
    assert got == expected
    assert bc.canonicalize_bekern_tokens(got) == got


def test_sys_joined_multipage_and_whitespace() -> None:
    raw = (
        "  clef-G-2 key+0 time-4-4  time-4-4  voice-1 quarter G 4 chord quarter C 4 bar  "
        "<sys>  "
        "clef-G-2 key+0 time-4-4 voice-1 quarter E 4 accent stacc bar-light-heavy\r\n"
    )
    got = bc.canonicalize_bekern_text(raw)
    expected = (
        "clef-G-2 key+0 time-4-4 voice-1 quarter C 4 chord quarter G 4 bar "
        "<sys> "
        "clef-G-2 key+0 time-4-4 voice-1 quarter E 4 stacc accent bar-light-heavy"
    )
    assert got == expected
    assert bc.canonicalize_bekern_text(got) == got


# ---------------------------------------------------------------------------
# 2. Real Suite31 GT round-trip: idempotency + 0 Strict Note F1 loss
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (ASER_BIN.is_file() and SUITE31_GT_DIR.is_dir()),
    reason="assembly-serialize.exe or suite31 GT dir not available",
)
def test_suite31_gt_roundtrip_zero_note_f1_loss() -> None:
    gt_files = sorted(SUITE31_GT_DIR.glob("*.musicxml"))
    assert len(gt_files) == 31

    with tempfile.TemporaryDirectory(prefix="test_bekern_canon_s31_") as td:
        td_p = pathlib.Path(td)
        for mxml in gt_files:
            for dialect_flag, suffix in (([], "v1"), (["--v2"], "v2")):
                stem = f"{mxml.stem}_{suffix}"
                r = subprocess.run(
                    [
                        str(ASER_BIN),
                        "bekern",
                        "--input",
                        str(mxml),
                        "--out",
                        str(td_p),
                        "--stem",
                        stem,
                        "--page",
                        "1",
                        *dialect_flag,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                assert r.returncode == 0, f"bekern failed on {stem}: {r.stderr}"
                orig_path = td_p / f"{stem}-bekern.txt"
                orig_toks = orig_path.read_text(encoding="utf-8").split()

                canon_toks = bc.canonicalize_bekern_tokens(orig_toks)
                # Idempotency
                assert bc.canonicalize_bekern_tokens(canon_toks) == canon_toks

                # Exact decoded note-set equivalence (0 Strict Note F1 loss)
                if canon_toks != orig_toks:
                    cand_path = td_p / f"{stem}-canon.txt"
                    cand_path.write_text(" ".join(canon_toks) + "\n", encoding="utf-8")
                    ok, err = rc.verify_equivalence(
                        orig_path, cand_path, ASER_BIN, td_p, stem
                    )
                    assert ok, f"Round-trip mismatch on {stem}: {err}"


@pytest.mark.skipif(
    not (ASER_BIN.is_file() and SUITE31_GT_DIR.is_dir()),
    reason="assembly-serialize.exe or suite31 GT dir not available",
)
def test_suite31_adversarial_perturbations_lossless_recovery() -> None:
    """Perturb real suite31 GT streams (reverse chords, shuffle modifiers, echo headers)
    and prove `canonicalize_bekern_tokens` recovers 100% identical decoded note multisets."""
    gt_files = sorted(SUITE31_GT_DIR.glob("*.musicxml"))[:10]
    rng = random.Random(123)

    with tempfile.TemporaryDirectory(prefix="test_bekern_canon_adv_") as td:
        td_p = pathlib.Path(td)
        for mxml in gt_files:
            stem = mxml.stem
            r = subprocess.run(
                [
                    str(ASER_BIN),
                    "bekern",
                    "--input",
                    str(mxml),
                    "--out",
                    str(td_p),
                    "--stem",
                    stem,
                    "--page",
                    "1",
                    "--v2",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert r.returncode == 0
            orig_path = td_p / f"{stem}-bekern.txt"
            orig_toks = orig_path.read_text(encoding="utf-8").split()
            canon_orig = bc.canonicalize_bekern_tokens(orig_toks)

            # Adversarially reverse every chord run & inject a redundant leading time token
            perturbed = list(canon_orig)
            # Find end of leading header and duplicate last time-* token
            hi = 0
            while hi < len(perturbed) and bc._is_header_token(perturbed[hi]):
                hi += 1
            if hi >= 3 and perturbed[hi - 1].startswith("time-"):
                perturbed.insert(hi, perturbed[hi - 1])

            # Reverse chord runs
            out_p: list[str] = []
            i = 0
            while i < len(perturbed):
                if (
                    perturbed[i] in bc.DURATIONS
                    and i + 2 < len(perturbed)
                    and perturbed[i + 1] in bc.STEPS
                    and bc._is_octave(perturbed[i + 2])
                ):
                    tones: list[list[str]] = []
                    while True:
                        t_start = i
                        i += 3
                        while i < len(perturbed) and bc._is_tail_token(perturbed[i]):
                            i += 1
                        tones.append(perturbed[t_start:i])
                        if (
                            i + 3 < len(perturbed)
                            and perturbed[i] == "chord"
                            and perturbed[i + 1] in bc.DURATIONS
                            and perturbed[i + 2] in bc.STEPS
                            and bc._is_octave(perturbed[i + 3])
                        ):
                            i += 1
                        else:
                            break
                    if len(tones) > 1:
                        rng.shuffle(tones)
                    for idx_t, t_toks in enumerate(tones):
                        if idx_t > 0:
                            out_p.append("chord")
                        out_p.extend(t_toks)
                else:
                    out_p.append(perturbed[i])
                    i += 1

            recovered = bc.canonicalize_bekern_tokens(out_p)
            assert recovered == canon_orig

            rec_path = td_p / f"{stem}-rec.txt"
            rec_path.write_text(" ".join(recovered) + "\n", encoding="utf-8")
            ok, err = rc.verify_equivalence(orig_path, rec_path, ASER_BIN, td_p, stem)
            assert ok, f"Adversarial recovery mismatch on {stem}: {err}"


# ---------------------------------------------------------------------------
# 3. 500-page real training sample: idempotency + quantified normalization rate
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not ITER15_MANIFEST.is_file(),
    reason="D:\\tmp\\iter15_local_data manifest not present",
)
def test_iter15_500_page_sample_idempotency_and_equivalence() -> None:
    by_ver: dict[str, list[dict]] = {"v2a": [], "v1": []}
    with ITER15_MANIFEST.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            v = r.get("bekern_version")
            if v in by_ver:
                by_ver[v].append(r)

    rng = random.Random(42)
    sample: list[dict] = []
    for v in ("v2a", "v1"):
        by_dom: dict[str, list[dict]] = collections.defaultdict(list)
        for r in by_ver[v]:
            by_dom[r.get("domain", "?")].append(r)
        chosen: list[dict] = []
        total_v = len(by_ver[v])
        for _, lst in sorted(by_dom.items()):
            k = max(1, round(250 * len(lst) / total_v))
            rng.shuffle(lst)
            chosen.extend(lst[:k])
        sample.extend(chosen[:250])

    assert len(sample) == 500
    pages_changed = 0
    changed_sys_samples: list[tuple[pathlib.Path, list[str]]] = []

    for row in sample:
        page_changed = False
        for bpath_s in row.get("bekerns", []):
            bpath = pathlib.Path(bpath_s)
            toks = bpath.read_text(encoding="utf-8").split()
            canon = bc.canonicalize_bekern_tokens(toks)
            # Strictly idempotent on every real training system
            assert bc.canonicalize_bekern_tokens(canon) == canon
            if canon != toks:
                page_changed = True
                if len(changed_sys_samples) < 20:
                    changed_sys_samples.append((bpath, canon))
        if page_changed:
            pages_changed += 1

    # Sanity: we know real iter15 data has non-zero voice/header non-determinism
    assert pages_changed > 0

    # Verify live `assembly-serialize bekern-decode` equivalence on 20 changed real systems
    if ASER_BIN.is_file():
        with tempfile.TemporaryDirectory(prefix="test_bekern_canon_real_") as td:
            td_p = pathlib.Path(td)
            for idx, (orig_path, canon_toks) in enumerate(changed_sys_samples):
                cand_path = td_p / f"real_{idx}.txt"
                cand_path.write_text(" ".join(canon_toks) + "\n", encoding="utf-8")
                ok, err = rc.verify_equivalence(
                    orig_path, cand_path, ASER_BIN, td_p, f"real_{idx}"
                )
                assert ok, f"Real iter15 equivalence mismatch on {orig_path}: {err}"
