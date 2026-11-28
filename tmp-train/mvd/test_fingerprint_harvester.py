#!/usr/bin/env python3
"""Unit tests for the CPU-only Fingerprint-First Forced Alignment Harvester."""

from __future__ import annotations

import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
import zipfile

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import fingerprint_harvester as fh


# ---------------------------------------------------------------------------
# Helpers to synthesize realistic multi-staff bekern & MusicXML test works
# ---------------------------------------------------------------------------

_STEPS = ["C", "D", "E", "F", "G", "A", "B"]
_MIDI_TO_STEP_ALT = [
    ("C", ""),
    ("C", "a1"),
    ("D", ""),
    ("E", "a-1"),
    ("E", ""),
    ("F", ""),
    ("F", "a1"),
    ("G", ""),
    ("A", "a-1"),
    ("A", ""),
    ("B", "a-1"),
    ("B", ""),
]
_DURS = ["quarter", "eighth", "eighth", "quarter", "half", "16th", "16th"]


def _midi_to_bekern_pitch(midi: int) -> list[str]:
    octv = (midi // 12) - 1
    step, alt = _MIDI_TO_STEP_ALT[midi % 12]
    return [step, str(octv), alt] if alt else [step, str(octv)]


def _make_synthetic_bekern(
    seed: int,
    *,
    n_staves: int = 2,
    n_measures: int = 24,
    notes_per_bar: int = 6,
    transpose_semitones: int = 0,
    corrupt_frac: float = 0.0,
    corrupt_seed: int = 999,
    measure_slice: tuple[int, int] | None = None,
    measures_per_sys: int = 4,
    drop_bar_indices: set[int] | None = None,
) -> list[str]:
    """Generate a deterministic, musically varied multi-staff bekern token stream."""
    rng = random.Random(seed)
    crng = random.Random(corrupt_seed)
    drop_bars = drop_bar_indices or set()

    # Pre-generate all measures deterministically from `seed`
    all_measures: list[list[list[tuple[int, str]]]] = []
    cur_pitches = [68 - 10 * s for s in range(n_staves)]
    for _ in range(n_measures):
        m_staves: list[list[tuple[int, str]]] = []
        for s in range(n_staves):
            st_notes: list[tuple[int, str]] = []
            for _ in range(notes_per_bar):
                step_delta = rng.choice([-5, -3, -2, -1, 1, 2, 3, 4, 5, 7])
                cur_pitches[s] = max(40, min(84, cur_pitches[s] + step_delta))
                dur = rng.choice(_DURS)
                st_notes.append((cur_pitches[s], dur))
            m_staves.append(st_notes)
        all_measures.append(m_staves)

    m0, m1 = measure_slice if measure_slice is not None else (0, n_measures)
    selected = all_measures[m0:m1]

    def _hdr() -> list[str]:
        toks: list[str] = []
        for s in range(n_staves):
            clef = "clef-G-2" if s == 0 else "clef-F-4"
            toks.extend([clef, "key+0", "time-4-4"])
        return toks

    out: list[str] = []
    for rel_i, m_staves in enumerate(selected):
        if rel_i > 0 and measures_per_sys > 0 and rel_i % measures_per_sys == 0:
            out.append(fh.SYS_TOKEN)
            out.extend(_hdr())
        elif rel_i == 0:
            out.extend(_hdr())

        for s, st_notes in enumerate(m_staves):
            out.append(f"voice-{s + 1}")
            for pitch, dur in st_notes:
                p = pitch + transpose_semitones
                d = dur
                if corrupt_frac > 0.0 and crng.random() < corrupt_frac:
                    mode = crng.random()
                    if mode < 0.65:
                        # Pitch substitution error
                        p += crng.choice([-4, -3, -2, 2, 3, 4])
                    elif mode < 0.85:
                        # Duration error
                        d = crng.choice(["whole", "32nd", "half"])
                    else:
                        # Note deletion
                        continue
                out.append(d)
                out.extend(_midi_to_bekern_pitch(p))
        if rel_i not in drop_bars:
            out.append("bar")
    return out


def _make_minimal_musicxml(n_staves: int = 2, n_measures: int = 12, seed: int = 7) -> str:
    """Create a valid multi-staff MusicXML string for parser tests."""
    rng = random.Random(seed)
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<score-partwise version="3.1">',
        "  <part-list><score-part id=\"P1\"><part-name>Piano</part-name></score-part></part-list>",
        '  <part id="P1">',
    ]
    cur = [67, 52, 60, 48]
    for m in range(1, n_measures + 1):
        lines.append(f'    <measure number="{m}">')
        if m == 1:
            lines.append("      <attributes>")
            lines.append("        <divisions>4</divisions>")
            lines.append("        <key><fifths>-1</fifths></key>")
            lines.append("        <time><beats>4</beats><beat-type>4</beat-type></time>")
            lines.append(f"        <staves>{n_staves}</staves>")
            lines.append("      </attributes>")
        for s in range(1, n_staves + 1):
            if s > 1:
                lines.append("      <backup><duration>16</duration></backup>")
            for _ in range(4):
                cur[s - 1] = max(40, min(80, cur[s - 1] + rng.choice([-3, -2, -1, 1, 2, 3, 5])))
                octv = (cur[s - 1] // 12) - 1
                step, alt = _MIDI_TO_STEP_ALT[cur[s - 1] % 12]
                alt_xml = ""
                if alt == "a1":
                    alt_xml = "<alter>1</alter>"
                elif alt == "a-1":
                    alt_xml = "<alter>-1</alter>"
                lines.append(
                    f"      <note><pitch><step>{step}</step>{alt_xml}<octave>{octv}</octave></pitch>"
                    f"<duration>4</duration><staff>{s}</staff></note>"
                )
        lines.append("    </measure>")
    lines.extend(["  </part>", "</score-partwise>"])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tests (compatible with both `python -m unittest` and `pytest`)
# ---------------------------------------------------------------------------


class TestFingerprintHarvester(unittest.TestCase):
    def test_exact_retrieval_under_40pct_corruption_and_transposition(self) -> None:
        """Target work must retrieve #1 with high margin & DTW alignment under 40% corruption + transposition."""
        idx = fh.SymbolicIndex()
        for w_i in range(25):
            toks = _make_synthetic_bekern(
                seed=1000 + w_i, n_staves=2, n_measures=32, notes_per_bar=8
            )
            score = fh.parse_bekern_tokens(toks, work_id=f"synth_work_{w_i:02d}")
            idx.add_score(score)

        assert idx.n_works == 25

        # Query: measures [8..20) of synth_work_13, transposed +5 semitones, 40% note corruption
        q_toks = _make_synthetic_bekern(
            seed=1013,
            n_staves=2,
            n_measures=32,
            notes_per_bar=8,
            transpose_semitones=5,
            corrupt_frac=0.40,
            corrupt_seed=42,
            measure_slice=(8, 20),
            measures_per_sys=4,
        )
        q_score = fh.parse_bekern_tokens(q_toks, work_id="noisy_page_13")

        ret, aln, dec = fh.harvest_page(
            q_score,
            idx,
            tau_align=0.35,
            tau_margin=0.15,
            check_eval_leakage=False,
        )

        assert ret.top1 is not None
        assert ret.top1.work_id == "synth_work_13"
        assert abs(ret.top1.measure_offset - 8) <= 1
        assert ret.margin >= 0.30, f"Expected strong margin, got {ret.margin}"
        assert aln is not None
        assert abs(aln.m_start - 8) <= 1
        assert abs(aln.m_end - 20) <= 1
        assert len(aln.system_breaks) == 3
        assert aln.align_score >= 0.35, f"Expected align_score >= 0.35, got {aln.align_score}"
        assert dec.accepted is True, f"Expected acceptance, got reasons={dec.reasons}"

    def test_wrong_piece_rejection(self) -> None:
        """An unindexed piece must be rejected by the margin / alignment gate."""
        idx = fh.SymbolicIndex()
        for w_i in range(20):
            toks = _make_synthetic_bekern(
                seed=2000 + w_i, n_staves=2, n_measures=24, notes_per_bar=6
            )
            idx.add_score(fh.parse_bekern_tokens(toks, work_id=f"catalog_work_{w_i:02d}"))

        # Query from completely unrelated seed not in index
        wrong_toks = _make_synthetic_bekern(
            seed=99999, n_staves=2, n_measures=12, notes_per_bar=6
        )
        q_score = fh.parse_bekern_tokens(wrong_toks, work_id="unindexed_manuscript")

        _, _, dec = fh.harvest_page(
            q_score,
            idx,
            tau_align=0.38,
            tau_margin=0.15,
            check_eval_leakage=False,
        )
        assert dec.accepted is False
        assert any(
            r.startswith("LOW_ALIGN_SCORE")
            or r.startswith("LOW_MARGIN")
            or r.startswith("LOW_SCORE")
            or r == "NO_CANDIDATE"
            for r in dec.reasons
        )

    def test_staff_count_mismatch_rejection(self) -> None:
        """Even with high fingerprint match, mismatched staff count (2 vs 3) MUST be rejected."""
        idx = fh.SymbolicIndex()
        gt_toks = _make_synthetic_bekern(seed=3001, n_staves=3, n_measures=20, notes_per_bar=6)
        gt_score = fh.parse_bekern_tokens(gt_toks, work_id="lied_3staff_modern")
        idx.add_score(gt_score)

        # Build 2-staff query containing the exact top 2 staves' notes (condensed layout)
        q_toks: list[str] = ["clef-G-2", "key+0", "time-4-4", "clef-F-4", "key+0", "time-4-4"]
        for m in gt_score.measures[:10]:
            for s in (0, 1):
                q_toks.append(f"voice-{s + 1}")
                for ev in m.staff_notes.get(s, []):
                    q_toks.append("quarter")
                    q_toks.extend(_midi_to_bekern_pitch(ev.midi))
            q_toks.append("bar")
        q_score = fh.parse_bekern_tokens(q_toks, work_id="condensed_2staff_ms")
        assert q_score.n_staves == 2

        ret, aln, dec = fh.harvest_page(
            q_score,
            idx,
            tau_align=0.30,
            tau_margin=0.10,
            check_eval_leakage=False,
        )
        assert ret.top1 is not None
        assert ret.top1.work_id == "lied_3staff_modern"
        assert dec.accepted is False
        assert any(r.startswith("STAFF_COUNT_MISMATCH") for r in dec.reasons)

    def test_eval_leakage_guard_blocks_suite31_and_northstar(self) -> None:
        """Eval-leakage guard must block suite31_hw, gate20, and OMR-NorthStar stems."""
        guard = fh.EvalLeakageGuard.from_files(
            extra_excluded_stems=["northstar-custom-piece-01"],
        )

        for stem in (
            "hw-mozart-k421",
            "gt-p1-hw-debussy-lindaraja",
            "mozart-k-421-mvt1",
            "hw-beethoven-op135",
            "northstar-custom-piece-01",
        ):
            leaked, reason = guard.check(stem)
            assert leaked is True, f"Expected leakage block for {stem}"
            assert reason

        leaked_ok, _ = guard.check("schubert-winterreise-d911-01-gute-nacht")
        assert leaked_ok is False

        idx = fh.SymbolicIndex()
        toks = _make_synthetic_bekern(seed=4040, n_staves=2, n_measures=16, notes_per_bar=6)
        idx.add_score(fh.parse_bekern_tokens(toks, work_id="hw-debussy-lindaraja"))
        q = fh.parse_bekern_tokens(toks, work_id="some_random_scan")

        _, _, dec = fh.harvest_page(
            q, idx, tau_align=0.35, tau_margin=0.15, leakage_guard=guard, check_eval_leakage=True
        )
        assert dec.accepted is False
        assert any(r.startswith("EVAL_LEAKAGE") for r in dec.reasons)

    def test_musicxml_mxl_and_bekern_jsonl_indexing_and_dedup(self) -> None:
        """Verify .musicxml, .mxl, digital bekern JSONL indexing, deduplication, and save/load."""
        with tempfile.TemporaryDirectory(prefix="fp_harv_test_") as td:
            tmp_path = Path(td)
            xml_str = _make_minimal_musicxml(n_staves=2, n_measures=10, seed=55)
            xml_path = tmp_path / "work_plain.musicxml"
            xml_path.write_text(xml_str, encoding="utf-8")

            mxl_str = _make_minimal_musicxml(n_staves=3, n_measures=10, seed=77)
            mxl_path = tmp_path / "work_zipped.mxl"
            with zipfile.ZipFile(mxl_path, "w") as zf:
                zf.writestr(
                    "META-INF/container.xml",
                    '<?xml version="1.0"?><container><rootfiles>'
                    '<rootfile full-path="score.xml"/></rootfiles></container>',
                )
                zf.writestr("score.xml", mxl_str)

            bk_toks = _make_synthetic_bekern(seed=88, n_staves=2, n_measures=10, notes_per_bar=6)
            bk_file = tmp_path / "page_s0-bekern.txt"
            bk_file.write_text(" ".join(bk_toks), encoding="utf-8")
            jsonl_path = tmp_path / "manifest.jsonl"
            rows = [
                {"image": "scoreA_Bravura-1.png", "bekerns": [str(bk_file)], "domain": "printed"},
                {"image": "scoreA_Gonville-1.png", "bekerns": [str(bk_file)], "domain": "printed"},
                {"image": "neg_blank_1.png", "bekerns": [str(bk_file)], "domain": "neg_control"},
            ]
            jsonl_path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

            idx = fh.build_index(
                musicxml_files=[xml_path, mxl_path],
                bekern_jsonls=[jsonl_path],
                allowed_jsonl_domains={"printed", "hw"},
                deduplicate=True,
            )
            assert idx.n_works == 3
            assert idx.skipped_duplicates == 1

            out_pkl = tmp_path / "index.pkl.gz"
            idx.save(out_pkl)
            idx2 = fh.SymbolicIndex.load(out_pkl)
            assert idx2.n_works == 3
            assert idx2.n_postings == idx.n_postings

            q = fh.parse_bekern_tokens(bk_toks, work_id="query_jsonl")
            ret, aln, dec = fh.harvest_page(q, idx2, check_eval_leakage=False)
            assert ret.top1 is not None
            assert ret.top1.work_id == "scoreA_Bravura-1"
            assert aln is not None and aln.align_score >= 0.95
            assert dec.accepted is True

    def test_duplicate_decode_rejection_and_synthetic_prefix_exclusion(self) -> None:
        """Duplicate decodes across >=2 distinct images must be rejected, and synthetic prefixes excluded by default."""
        with tempfile.TemporaryDirectory(prefix="fp_harv_synth_") as td:
            tmp_path = Path(td)
            bk_real = _make_synthetic_bekern(seed=501, n_staves=2, n_measures=12, notes_per_bar=6)
            bk_synth1 = _make_synthetic_bekern(seed=502, n_staves=2, n_measures=12, notes_per_bar=6)
            bk_synth2 = _make_synthetic_bekern(seed=503, n_staves=2, n_measures=12, notes_per_bar=6)
            bk_synth3 = _make_synthetic_bekern(seed=504, n_staves=2, n_measures=12, notes_per_bar=6)

            f_real = tmp_path / "real_s0.txt"
            f_s1 = tmp_path / "s1.txt"
            f_s2 = tmp_path / "s2.txt"
            f_s3 = tmp_path / "s3.txt"
            f_real.write_text(" ".join(bk_real), encoding="utf-8")
            f_s1.write_text(" ".join(bk_synth1), encoding="utf-8")
            f_s2.write_text(" ".join(bk_synth2), encoding="utf-8")
            f_s3.write_text(" ".join(bk_synth3), encoding="utf-8")

            jsonl_path = tmp_path / "manifest.jsonl"
            rows = [
                {"image": "openscore_bach_bwv1001.png", "bekerns": [str(f_real)], "domain": "printed"},
                {"image": "hwsim_00123.png", "bekerns": [str(f_s1)], "domain": "hw"},
                {"image": "hw_sim_00456.png", "bekerns": [str(f_s2)], "domain": "hw"},
                {"image": "synth_page_007.png", "bekerns": [str(f_s3)], "domain": "printed"},
            ]
            jsonl_path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

            # Default build_index excludes hwsim_, hw_sim_, synth_
            idx_clean = fh.build_index(bekern_jsonls=[jsonl_path])
            assert idx_clean.n_works == 1
            assert idx_clean.skipped_synthetic == 3
            assert "openscore_bach_bwv1001" in idx_clean.work_id_to_idx

            # Disabling exclude_prefixes indexes all 4, and retrieval-time exclude_prefixes filters them out
            idx_all = fh.build_index(bekern_jsonls=[jsonl_path], exclude_prefixes=())
            assert idx_all.n_works == 4
            q_synth = fh.parse_bekern_tokens(bk_synth1, work_id="q_synth", source_path="img_synth.jpg")
            ret_filt = fh.retrieve_best_fit(
                q_synth, idx_all, exclude_prefixes=fh.DEFAULT_SYNTHETIC_PREFIXES
            )
            assert ret_filt.top1 is None or not ret_filt.top1.work_id.startswith("hwsim_")

        # Batch duplicate-decode hallucination guard: 2 distinct images with identical decode -> both rejected
        idx = fh.SymbolicIndex()
        t_a = _make_synthetic_bekern(seed=601, n_staves=2, n_measures=16, notes_per_bar=6)
        t_b = _make_synthetic_bekern(seed=602, n_staves=2, n_measures=16, notes_per_bar=6)
        idx.add_score(fh.parse_bekern_tokens(t_a, work_id="real_work_a"))
        idx.add_score(fh.parse_bekern_tokens(t_b, work_id="real_work_b"))

        q1 = fh.parse_bekern_tokens(t_a, work_id="rism_001", source_path="rism_001.jpg")
        q2 = fh.parse_bekern_tokens(t_a, work_id="rism_002", source_path="rism_002.jpg")
        q_ok = fh.parse_bekern_tokens(t_b, work_id="rism_003", source_path="rism_003.jpg")

        _, _, d1 = fh.harvest_page(q1, idx, check_eval_leakage=False)
        _, _, d2 = fh.harvest_page(q2, idx, check_eval_leakage=False)
        _, _, d3 = fh.harvest_page(q_ok, idx, check_eval_leakage=False)
        assert d1.accepted and d2.accepted and d3.accepted

        fh.filter_attractors([d1, d2, d3], max_queries_per_work=2, max_duplicate_decodes=1)
        assert d1.accepted is False
        assert "HALLUCINATION_DUPLICATE_DECODE" in d1.reasons
        assert d2.accepted is False
        assert "HALLUCINATION_DUPLICATE_DECODE" in d2.reasons
        assert d3.accepted is True

    def test_attractor_work_rejection(self) -> None:
        """A single work_id matched by >max_queries_per_work distinct images must be rejected."""
        idx = fh.SymbolicIndex()
        w_attr = _make_synthetic_bekern(seed=701, n_staves=2, n_measures=24, notes_per_bar=8)
        w_good = _make_synthetic_bekern(seed=702, n_staves=2, n_measures=24, notes_per_bar=8)
        idx.add_score(fh.parse_bekern_tokens(w_attr, work_id="attractor_work"))
        idx.add_score(fh.parse_bekern_tokens(w_good, work_id="genuine_work"))

        # 3 distinct query images with distinct noisy decodes all matching attractor_work
        decisions = []
        for k, (ms, me, cseed) in enumerate([(0, 10, 11), (2, 12, 22), (4, 14, 33)]):
            qt = _make_synthetic_bekern(
                seed=701,
                n_staves=2,
                n_measures=24,
                notes_per_bar=8,
                corrupt_frac=0.15,
                corrupt_seed=cseed,
                measure_slice=(ms, me),
            )
            q = fh.parse_bekern_tokens(qt, work_id=f"rism_attr_{k}", source_path=f"rism_attr_{k}.jpg")
            _, _, dec = fh.harvest_page(q, idx, tau_align=0.35, tau_margin=0.15, check_eval_leakage=False)
            assert dec.accepted is True and dec.work_id == "attractor_work"
            decisions.append(dec)

        # 2 distinct query images matching genuine_work (<= max_queries_per_work=2) stay accepted
        for k, (ms, me, cseed) in enumerate([(0, 10, 44), (12, 22, 55)]):
            qt = _make_synthetic_bekern(
                seed=702,
                n_staves=2,
                n_measures=24,
                notes_per_bar=8,
                corrupt_frac=0.10,
                corrupt_seed=cseed,
                measure_slice=(ms, me),
            )
            q = fh.parse_bekern_tokens(qt, work_id=f"rism_good_{k}", source_path=f"rism_good_{k}.jpg")
            _, _, dec = fh.harvest_page(q, idx, tau_align=0.35, tau_margin=0.15, check_eval_leakage=False)
            assert dec.accepted is True and dec.work_id == "genuine_work"
            decisions.append(dec)

        fh.filter_attractors(decisions, max_queries_per_work=2, max_duplicate_decodes=1)

        for d in decisions[:3]:
            assert d.accepted is False
            assert "HALLUCINATION_ATTRACTOR_WORK" in d.reasons
        for d in decisions[3:]:
            assert d.accepted is True


if __name__ == "__main__":
    unittest.main()
