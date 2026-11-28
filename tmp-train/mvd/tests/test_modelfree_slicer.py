#!/usr/bin/env python3
"""Fast, self-contained CPU unit tests for modelfree_slicer (v2).

ZERO torch imports, ZERO D:\\tmp dependencies.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import zipfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np
from PIL import Image
import pytest

MVD_DIR = Path(__file__).resolve().parents[1]
if str(MVD_DIR) not in sys.path:
    sys.path.insert(0, str(MVD_DIR))

import fingerprint_harvester as fh  # noqa: E402
import modelfree_slicer as mfs  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers for synthetic MusicXML & synthetic 2D symbol-ink profiles
# ---------------------------------------------------------------------------

def _make_sample_musicxml(
    n_measures: int = 8,
    *,
    mid_measure_attr_at: int = 2,
    two_staves: bool = False,
    part2_measures: int | None = None,
) -> ET.Element:
    root = ET.Element("score-partwise", {"version": "3.1"})
    pl = ET.SubElement(root, "part-list")
    sp1 = ET.SubElement(pl, "score-part", {"id": "P1"})
    ET.SubElement(sp1, "part-name").text = "Piano" if two_staves else "Violin"
    if part2_measures is not None:
        sp2 = ET.SubElement(pl, "score-part", {"id": "P2"})
        ET.SubElement(sp2, "part-name").text = "Cello"

    def _fill_part(p_el: ET.Element, count: int, multi_st: bool) -> None:
        for m in range(count):
            m_el = ET.SubElement(p_el, "measure", {"number": str(m + 1)})
            if m == 0:
                att = ET.SubElement(m_el, "attributes")
                ET.SubElement(att, "divisions").text = "4"
                k = ET.SubElement(att, "key")
                ET.SubElement(k, "fifths").text = "2"
                t = ET.SubElement(att, "time")
                ET.SubElement(t, "beats").text = "4"
                ET.SubElement(t, "beat-type").text = "4"
                if multi_st:
                    ET.SubElement(att, "staves").text = "2"
                    c1 = ET.SubElement(att, "clef", {"number": "1"})
                    ET.SubElement(c1, "sign").text = "G"
                    ET.SubElement(c1, "line").text = "2"
                    c2 = ET.SubElement(att, "clef", {"number": "2"})
                    ET.SubElement(c2, "sign").text = "F"
                    ET.SubElement(c2, "line").text = "4"
                else:
                    c1 = ET.SubElement(att, "clef", {"number": "1"})
                    ET.SubElement(c1, "sign").text = "G"
                    ET.SubElement(c1, "line").text = "2"
                tr = ET.SubElement(att, "transpose")
                ET.SubElement(tr, "diatonic").text = "-1"
                ET.SubElement(tr, "chromatic").text = "-2"

            # Add notes
            n1 = ET.SubElement(m_el, "note")
            p1 = ET.SubElement(n1, "pitch")
            ET.SubElement(p1, "step").text = "C"
            ET.SubElement(p1, "octave").text = "5"
            ET.SubElement(n1, "duration").text = "8"
            if multi_st:
                ET.SubElement(n1, "staff").text = "1"

            if m == mid_measure_attr_at:
                # Mid-measure clef change AFTER a note!
                mid_att = ET.SubElement(m_el, "attributes")
                mc = ET.SubElement(mid_att, "clef", {"number": "1"})
                ET.SubElement(mc, "sign").text = "C"
                ET.SubElement(mc, "line").text = "3"

            if multi_st:
                n2 = ET.SubElement(m_el, "note")
                p2 = ET.SubElement(n2, "pitch")
                ET.SubElement(p2, "step").text = "G"
                ET.SubElement(p2, "octave").text = "3"
                ET.SubElement(n2, "duration").text = "8"
                ET.SubElement(n2, "staff").text = "2"

    p1 = ET.SubElement(root, "part", {"id": "P1"})
    _fill_part(p1, n_measures, two_staves)
    if part2_measures is not None:
        p2 = ET.SubElement(root, "part", {"id": "P2"})
        _fill_part(p2, part2_measures, False)
    return root


def _make_distinct_measure_bank(
    n_measures: int, n_staves: int = 2, bar_w: int = 80, staff_h: int = 40, seed: int = 42
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Build a synthetic 2D symbol-ink strip where every measure & staff has unique rhythmic/pitch patterns."""
    rng = np.random.default_rng(seed)
    h = n_staves * staff_h
    w = n_measures * bar_w
    ink = np.zeros((h, w), dtype=np.float32)
    bands = [(s * staff_h, (s + 1) * staff_h) for s in range(n_staves)]
    for m in range(n_measures):
        x0 = m * bar_w
        for s in range(n_staves):
            y0 = s * staff_h
            # Place 3-6 distinct noteheads/stems per measure per staff
            n_ev = int(rng.integers(3, 7))
            xs = np.sort(rng.choice(np.arange(8, bar_w - 8, 6), size=n_ev, replace=False))
            for xi in xs:
                yc = int(rng.integers(y0 + 6, y0 + staff_h - 6))
                ink[yc - 3 : yc + 4, x0 + xi - 2 : x0 + xi + 3] = 1.0
                # Vertical stem
                if yc > y0 + staff_h // 2:
                    ink[max(y0 + 2, yc - 14) : yc, x0 + xi + 2] = 1.0
                else:
                    ink[yc : min(y0 + staff_h - 2, yc + 14), x0 + xi - 2] = 1.0
    return ink, bands


# ---------------------------------------------------------------------------
# 1. Model-Free Constraint & CV / Barline Primitives
# ---------------------------------------------------------------------------

def test_no_torch_imported() -> None:
    assert "torch" not in sys.modules


def test_cluster_barline_cols_two_stage() -> None:
    # sys_w = 1000 -> tight_gap = 12, max_span = 25, min_bar_sep = 32
    # Cluster 1: clean barline at 200..204 (span 4 -> accepted, centroid 202)
    # Cluster 2: wide gutter at 400..440 (span 40 > 25 -> discarded!)
    # Cluster 3 & 4: two close barlines at 700..703 and 730..733 (sep 30 < 32 -> merged!)
    cols = np.array(
        list(range(200, 205))
        + list(range(400, 441, 4))
        + list(range(700, 704))
        + list(range(730, 734)),
        dtype=np.int64,
    )
    bars, raw_n = mfs.cluster_barline_cols(cols, sys_w=1000)
    assert raw_n == 4
    assert len(bars) == 2
    assert abs(bars[0] - 202) <= 1
    assert abs(bars[1] - 716) <= 2


def test_dedup_bar_xs_unsorted() -> None:
    seen = [100, 300]
    # Unsorted duplicate near 100 MUST be rejected even though 105 is far from seen[-1]=300
    assert not mfs._dedup_bar_xs_unsorted(105, seen, min_sep=20)
    assert not mfs._dedup_bar_xs_unsorted(285, seen, min_sep=20)
    assert mfs._dedup_bar_xs_unsorted(200, seen, min_sep=20)


def test_recursive_cluster_splitting_three_staves() -> None:
    # 3 merged staves (15 lines, d_line=10, inter-staff gap=18)
    s1 = [10, 20, 30, 40, 50]
    s2 = [68, 78, 88, 98, 108]
    s3 = [126, 136, 146, 156, 166]
    merged = [s1 + s2 + s3]
    out = mfs._split_merged_staff_clusters(merged, d_line=10.0)
    assert out == [s1, s2, s3]


def test_horizontal_runs_mask_morph_open() -> None:
    ink = np.zeros((5, 30), dtype=bool)
    ink[1, 2:6] = True   # length 4 < 9 -> removed
    ink[3, 5:20] = True  # length 15 >= 9 -> kept
    out = mfs._horizontal_runs_mask(ink, min_run=9)
    assert not out[1].any()
    assert out[3, 5:20].all()
    assert not out[3, :5].any()


def test_trim_dark_scanner_borders() -> None:
    gray = np.full((200, 200), 230, dtype=np.uint8)
    gray[:12, :] = 10
    gray[-10:, :] = 15
    gray[:, :14] = 12
    gray[:, -8:] = 20
    trimmed = mfs._trim_dark_scanner_borders(gray)
    assert trimmed.shape == (178, 178)
    assert float(trimmed.min()) >= 200.0


def test_barline_detector_slanted_and_1staff() -> None:
    # 2-staff system with 2 slanted internal barlines (~8 deg)
    h, w = 220, 1000
    ink_vert = np.zeros((h, w), dtype=bool)
    for bx in (350, 680):
        for y in range(25, 195):
            dx = int(round((y - 110) * np.tan(np.deg2rad(7.0))))
            x = bx + dx
            ink_vert[y, max(0, x - 1) : min(w, x + 2)] = True
    sys_crop = {
        "y_top": 10,
        "y_bot": 210,
        "x_left": 50,
        "x_right": 950,
        "staves": [
            {"y_top": 30, "y_bot": 75, "x_left": 50, "x_right": 950},
            {"y_top": 140, "y_bot": 185, "x_left": 50, "x_right": 950},
        ],
    }
    m_est, bar_xs, bar_conf = mfs.estimate_system_measures_from_barlines(ink_vert, sys_crop)
    assert len(bar_xs) == 2
    assert m_est == 3
    assert bar_conf >= 0.5

    # 1-staff system with a note stem + adjacent notehead blob MUST have bar_conf == 0.0
    ink_1 = np.zeros((100, 1000), dtype=bool)
    ink_1[20:80, 400:402] = True  # stem
    ink_1[55:75, 396:400] = True  # adjacent notehead blob
    sc1 = {
        "y_top": 10,
        "y_bot": 90,
        "x_left": 50,
        "x_right": 950,
        "staves": [{"y_top": 20, "y_bot": 80, "x_left": 50, "x_right": 950}],
    }
    _, bar_xs1, bar_conf1 = mfs.estimate_system_measures_from_barlines(ink_1, sc1)
    assert bar_conf1 == 0.0
    assert len(bar_xs1) == 0


# ---------------------------------------------------------------------------
# 2. MusicXML Indexing, Mid-Measure <attributes>, DTD Order, Partbook Extract
# ---------------------------------------------------------------------------

def test_mid_measure_attributes_dtd_and_print_object_no() -> None:
    root = _make_sample_musicxml(n_measures=6, mid_measure_attr_at=2, part2_measures=4)
    indexed = mfs.index_musicxml_root(root)
    assert indexed.n_measures == 4  # min across non-empty parts
    assert indexed.part_measure_mismatch is True
    assert indexed.part_measure_counts == [6, 4]

    # Slice starting at measure index 2 (which has a mid-measure <attributes> AFTER <note>)
    sliced_str = mfs.slice_musicxml_measures(indexed, 2, 4)
    s_root = ET.fromstring(sliced_str)
    p1 = s_root.find("part[@id='P1']")
    assert p1 is not None
    m0 = p1.findall("measure")[0]
    children = [mfs._strip_ns(c.tag) for c in m0]
    # New <attributes> MUST be inserted BEFORE <note>, NOT merged into the mid-measure <attributes>!
    assert children[0] == "attributes"
    assert children.count("attributes") == 2
    assert children.index("attributes") < children.index("note")

    # Verify MusicXML 3.1 DTD order inside carried-forward <attributes>
    opening_attr = m0[0]
    attr_tags = [mfs._strip_ns(a.tag) for a in opening_attr]
    assert attr_tags == ["divisions", "key", "time", "clef", "transpose"]

    # Carried-forward <time> MUST have print-object="no" when m_start > 0
    time_el = opening_attr.find("time")
    assert time_el is not None
    assert time_el.attrib.get("print-object") == "no"

    # When m_start == 0, <time> should NOT have print-object="no"
    sliced_0 = ET.fromstring(mfs.slice_musicxml_measures(indexed, 0, 2))
    t0 = sliced_0.find("./part[@id='P1']/measure/attributes/time")
    assert t0 is not None
    assert t0.attrib.get("print-object") is None


def test_extract_single_staff_musicxml_grand_staff() -> None:
    root = _make_sample_musicxml(n_measures=4, two_staves=True)
    indexed = mfs.index_musicxml_root(root)
    assert indexed.total_staves == 2

    s1 = mfs.extract_single_staff_musicxml(indexed, staff_idx=1)  # bass staff
    assert s1 is not None
    assert s1.total_staves == 1
    assert s1.n_measures == 4
    xml_out = ET.fromstring(mfs.slice_musicxml_measures(s1, 0, 2))
    clefs = xml_out.findall(".//clef")
    assert len(clefs) == 1
    assert clefs[0].findtext("sign") == "F"
    assert clefs[0].attrib.get("number") == "1"
    notes = xml_out.findall(".//note")
    assert all(n.findtext("pitch/step") == "G" for n in notes)


# ---------------------------------------------------------------------------
# 3. Verovio SVG Parsing & Staff-Line Stripping
# ---------------------------------------------------------------------------

def test_parse_and_strip_verovio_svg() -> None:
    svg = """<svg viewBox="0 0 200 100" xmlns="http://www.w3.org/2000/svg">
      <svg class="definition-scale" viewBox="0 0 2000 1000">
        <g class="page-margin" transform="translate(50, 50)">
          <g class="measure" id="m1">
            <g class="staff" id="s1">
              <path d="M100 100 L900 100" stroke-width="10"/>
              <path d="M100 140 L900 140" stroke-width="10"/>
              <path d="M100 180 L900 180" stroke-width="10"/>
              <path d="M100 220 L900 220" stroke-width="10"/>
              <path d="M100 260 L900 260" stroke-width="10"/>
              <g class="layer"><circle cx="300" cy="180" r="20"/></g>
            </g>
          </g>
          <g class="measure" id="m2">
            <g class="staff" id="s2">
              <path d="M900 100 L1800 100" stroke-width="10"/>
              <path d="M900 140 L1800 140" stroke-width="10"/>
              <path d="M900 180 L1800 180" stroke-width="10"/>
              <path d="M900 220 L1800 220" stroke-width="10"/>
              <path d="M900 260 L1800 260" stroke-width="10"/>
              <g class="layer"><circle cx="1200" cy="140" r="20"/></g>
            </g>
          </g>
        </g>
      </svg>
    </svg>"""
    parsed = mfs._parse_and_strip_verovio_svg(svg, n_staves=1)
    assert parsed is not None
    stripped_svg, w_def, h_def, tx, ty, m_xs, s_ys = parsed
    assert (w_def, h_def, tx, ty) == (2000.0, 1000.0, 50.0, 50.0)
    assert m_xs == [(100.0, 900.0), (900.0, 1800.0)]
    assert s_ys == [(100.0, 260.0)]
    assert "<path" not in stripped_svg  # horizontal staff lines stripped!

    # Staff count mismatch MUST reject (return None, never silent uniform split!)
    assert mfs._parse_and_strip_verovio_svg(svg, n_staves=2) is None


# ---------------------------------------------------------------------------
# 4. Itakura Slope-Constrained DTW, Global 2-Best DP & Strict 3-Tier Gate
# ---------------------------------------------------------------------------

def test_itakura_dtw_identity_stretch_and_degenerate_warp() -> None:
    rng = np.random.default_rng(7)
    N = 200
    raw = rng.normal(size=(N, 10)).astype(np.float32)
    raw = cv2.GaussianBlur(raw, (1, 5), 1.2)
    F = raw / (np.linalg.norm(raw, axis=1, keepdims=True) + 1e-6)

    # Identical -> score == 1.0, warp_frac == 0.0
    sc, path, warp_frac = mfs.compare_system_profiles_dtw(F, F)
    assert abs(sc - 1.0) < 1e-4
    assert warp_frac == 0.0
    assert path[0] == (0, 0) and path[-1] == (N - 1, N - 1)

    # Smooth non-linear time warp within ±10% -> high score, accepted
    t = np.linspace(0, N - 1, N)
    t_warped = np.clip(t + 6.0 * np.sin(2.0 * np.pi * t / N), 0, N - 1)
    F_w = np.stack([np.interp(t_warped, t, F[:, d]) for d in range(F.shape[1])], axis=1).astype(np.float32)
    F_w /= np.linalg.norm(F_w, axis=1, keepdims=True) + 1e-6
    sc_w, _, wf_w = mfs.compare_system_profiles_dtw(F, F_w)
    assert sc_w > 0.75
    assert wf_w <= 0.10

    # Degenerate large shift exceeding max_mean_warp_frac -> rejected (-1.0)
    F_shift = np.roll(F, shift=30, axis=0)
    sc_deg, _, wf_deg = mfs.compare_system_profiles_dtw(F, F_shift, band=40, max_mean_warp_frac=0.05)
    assert sc_deg == -1.0
    assert wf_deg > 0.05 or sc_deg == -1.0


def test_global_2best_dp_and_3tier_gates() -> None:
    n_measures = 24
    n_staves = 2
    bar_w = 80
    staff_h = 40
    bank_ink, bands = _make_distinct_measure_bank(n_measures, n_staves, bar_w, staff_h, seed=123)
    indexed = mfs.index_musicxml_root(_make_sample_musicxml(n_measures=n_measures, two_staves=True))

    def render_fn(_idx: mfs.IndexedMusicXML, m0: int, m1: int, _ns: int):
        sub = bank_ink[:, m0 * bar_w : m1 * bar_w]
        return sub, bands, False

    # Construct a synthetic 2-system manuscript page matching [0, 4) and [4, 7)
    def _make_vis_from_ranges(ranges: list[tuple[int, int]], scramble_staff1: bool = False):
        sys_h = n_staves * staff_h
        page_w = 600
        page_h = len(ranges) * (sys_h + 40) + 40
        deskewed = np.full((page_h, page_w), 240, dtype=np.uint8)
        ink_vert = np.zeros((page_h, page_w), dtype=bool)
        sys_crops = []
        for s_i, (m0, m1) in enumerate(ranges):
            sub = bank_ink[:, m0 * bar_w : m1 * bar_w].copy()
            if scramble_staff1:
                # Replace staff 1 with unrelated measures from end of piece -> fails Tier 2
                sub[staff_h:, :] = bank_ink[staff_h:, 18 * bar_w : (18 + (m1 - m0)) * bar_w]
            # Resize horizontally into page width with 9% left preamble pad
            content_w = 460
            pre_w = int(round(content_w * 0.09 / 0.91))
            sub_rs = cv2.resize(sub, (content_w, sys_h), interpolation=cv2.INTER_AREA)
            y0 = 20 + s_i * (sys_h + 40)
            x0 = 40
            # Draw dark ink on white background
            deskewed[y0 : y0 + sys_h, x0 + pre_w : x0 + pre_w + content_w] = np.clip(
                240.0 - 200.0 * sub_rs, 20, 240
            ).astype(np.uint8)
            sys_crops.append({
                "y_top": y0,
                "y_bot": y0 + sys_h,
                "x_left": x0,
                "x_right": x0 + pre_w + content_w,
                "staves": [
                    {"y_top": y0 + 6, "y_bot": y0 + staff_h - 6, "x_left": x0, "x_right": x0 + pre_w + content_w},
                    {"y_top": y0 + staff_h + 6, "y_bot": y0 + 2 * staff_h - 6, "x_left": x0, "x_right": x0 + pre_w + content_w},
                ],
            })
        return {"deskewed": deskewed, "ink_vert": ink_vert, "system_crops": sys_crops}

    # Case A: Clean matching page [0, 4), [4, 7) -> ACCEPTED by all 3 tiers!
    vis_ok = _make_vis_from_ranges([(0, 4), (4, 7)])
    res_ok = mfs.slice_page_modelfree(vis_ok, indexed, n_staves=2, render_fn=render_fn)
    assert res_ok["accepted"] is True, f"Expected accept, got {res_ok['reason']}"
    assert res_ok["system_breaks"] == [[0, 4], [4, 7]]
    assert res_ok["dp_margin"] >= 0.025
    assert all(d["z_decoy"] >= 2.5 for d in res_ok["sys_details"])
    assert all(d["min_staff_r"] >= 0.22 for d in res_ok["sys_details"])

    # Case B: Multi-page chaining via m_start_candidates -> matches [7, 10), [10, 13)
    vis_p2 = _make_vis_from_ranges([(7, 10), (10, 13)])
    res_p2 = mfs.slice_page_modelfree(
        vis_p2, indexed, n_staves=2, m_start_candidates=(6, 7, 8), render_fn=render_fn
    )
    assert res_p2["accepted"] is True, f"Expected page-2 accept, got {res_p2['reason']}"
    assert res_p2["system_breaks"] == [[7, 10], [10, 13]]

    # Case C: Scrambled staff 1 -> REJECTED by 3-tier gate!
    vis_bad_st = _make_vis_from_ranges([(0, 4), (4, 7)], scramble_staff1=True)
    res_bad_st = mfs.slice_page_modelfree(vis_bad_st, indexed, n_staves=2, render_fn=render_fn)
    assert res_bad_st["accepted"] is False
    assert any(tag in res_bad_st["reason"] for tag in ("tier1_", "tier2_", "tier3_"))

    # Case D: Repetitive/identical measures across the whole piece -> REJECTED by Tier 1 decoy / Tier 3 margin!
    rep_bar = bank_ink[:, 0:bar_w]
    rep_bank = np.tile(rep_bar, (1, n_measures))

    def rep_render_fn(_idx: mfs.IndexedMusicXML, m0: int, m1: int, _ns: int):
        return rep_bank[:, m0 * bar_w : m1 * bar_w], bands, False

    res_rep = mfs.slice_page_modelfree(vis_ok, indexed, n_staves=2, render_fn=rep_render_fn)
    assert res_rep["accepted"] is False


# ---------------------------------------------------------------------------
# 5. Export Uniformity (No Mixed), Grammar Rejection, Measure-Count Assertion
# ---------------------------------------------------------------------------

def test_count_bekern_measures() -> None:
    toks = [
        "clef-G-2", "quarter", "C", "4", "bar-repeat-right",
        "bar-repeat-left", "bar-rest", "bar",
        "quarter", "D", "4", "bar-double",
        "quarter", "E", "4", "bar-light-heavy",
    ]
    assert mfs.count_bekern_measures(toks) == 4


def test_export_page_uniform_v2a_fallback_v1_and_grammar_reject(tmp_path: Path) -> None:
    xml_p = tmp_path / "dummy.musicxml"
    xml_p.write_text("<score-partwise/>", encoding="utf-8")

    # v2a succeeds on s0, fails on s1 -> MUST fall back to uniform v1 (never mixed!)
    def fake_export(
        _xml, out_dir: Path, page_stem: str, sys_idx: int, _ms: int, _me: int, *, version: str, **_kw
    ):
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"{page_stem}_s{sys_idx}-bekern.txt"
        if version == "v2a" and sys_idx == 1:
            return None, "oov"
        p.write_text(f"ver_{version}_s{sys_idx}", encoding="utf-8")
        return p, "ok"

    paths, ver, msg = mfs.export_page_uniform_bekern(
        xml_p,
        tmp_path / "bk",
        "p1",
        [[0, 3], [3, 6]],
        export_fn=fake_export,
        grammar_fn=lambda _ps: (True, "ok"),
    )
    assert ver == "v1"
    assert msg == "ok"
    assert paths is not None and len(paths) == 2
    assert all("ver_v1_" in Path(p).read_text(encoding="utf-8") for p in paths)

    # Hard reject when grammar_fn returns False -> cleans up exported files
    paths_rej, ver_rej, err_rej = mfs.export_page_uniform_bekern(
        xml_p,
        tmp_path / "bk2",
        "p2",
        [[0, 3], [3, 6]],
        export_fn=fake_export,
        grammar_fn=lambda _ps: (False, "grammar_broken"),
    )
    assert paths_rej is None
    assert ver_rej == ""
    assert "grammar_reject" in err_rej
    assert not list((tmp_path / "bk2").glob("*.txt"))


# ---------------------------------------------------------------------------
# 6. EvalLeakageGuard Cross-Field Check, CorpusIndex, Atomic Writes & Cache
# ---------------------------------------------------------------------------

def test_eval_leakage_guard_cross_field_combined() -> None:
    guard = fh.EvalLeakageGuard()
    guard._register_slug("mozart-k458-mvt1")
    # Composer in work_id, catalog number in title -> caught by combined string!
    row = {
        "rism_id": "999999999",
        "image_path": "page_001.jpg",
        "work_id": "openscore_mozart_quartet",
        "work_source_path": "sq01.mxl",
        "resolved_source_path": "D:/scores/sq01.mxl",
        "title": "String Quartet in B-flat major, KV 458",
    }
    leaked, reason = mfs.check_eval_overlap(row, [("test", guard)])
    assert leaked is True
    assert "eval_catalog" in reason


def test_corpus_index_ambiguity_abstention(tmp_path: Path) -> None:
    sq_root = tmp_path / "sq"
    (sq_root / "a").mkdir(parents=True)
    (sq_root / "b").mkdir(parents=True)
    (sq_root / "a" / "sq01.mxl").write_bytes(b"zip1")
    (sq_root / "b" / "sq01.mxl").write_bytes(b"zip2")  # duplicate -> ambiguous!
    (sq_root / "a" / "sq02.mxl").write_bytes(b"zip3")  # unique!

    idx = mfs.CorpusIndex({"openscore_sq": sq_root}).build()
    assert idx.resolve("work_sq01_foo", "foo.png") == ("", "")
    path_ok, ext_ok = idx.resolve("work_sq02_bar", "bar.png")
    assert ext_ok == ".mxl" and path_ok.endswith("sq02.mxl")


def test_atomic_writes_hash_keyed_xml_cache_and_lru(tmp_path: Path) -> None:
    # Atomic write
    target = tmp_path / "sub" / "out.txt"
    mfs.atomic_write_text(target, "hello-world")
    assert target.read_text(encoding="utf-8") == "hello-world"
    assert not list(target.parent.glob("*.tmp.*"))

    # Hash-keyed .mxl -> .musicxml conversion
    mxl_path = tmp_path / "piece.mxl"
    xml_bytes = ET.tostring(_make_sample_musicxml(n_measures=2), encoding="utf-8")
    with zipfile.ZipFile(mxl_path, "w") as zf:
        zf.writestr("score.musicxml", xml_bytes)
    cache_dir = tmp_path / "xml_cache"
    out_xml = mfs.convert_to_musicxml(mxl_path, cache_dir)
    assert out_xml is not None and out_xml.is_file()
    assert out_xml.name.startswith("piece_") and out_xml.name.endswith(".musicxml")
    assert len(out_xml.stem.split("_")[-1]) == 12  # 12-char sha1 prefix

    # LRU bound <= 32
    lru = mfs.IndexedMusicXMLCache(max_size=3)
    for i in range(5):
        p = tmp_path / f"w{i}.musicxml"
        p.write_bytes(xml_bytes)
        lru.get(p)
    assert len(lru) == 3


# ---------------------------------------------------------------------------
# 7. Barline Round 2 Regression Tests
# ---------------------------------------------------------------------------

def test_extract_1d_profiles_tacet_norm_floor_and_clamping() -> None:
    # Staff 0 is active with strong notes; Staff 1 is nearly blank/tacet with tiny noise
    h, w = 80, 200
    ink = np.zeros((h, w), dtype=np.float32)
    for x in range(20, 180, 20):
        ink[12:28, x : x + 6] = 1.0
    ink[55:57, 100:102] = 0.05  # tiny speck on tacet staff 1
    # Include an out-of-bounds band to verify clamping doesn't crash or produce NaN
    bands = [(0, 40), (40, 120)]
    F, active = mfs.extract_1d_profiles(ink, bands, n_bars=4, target_w=192, mask_preamble_frac=0.0)
    assert F.shape == (192, 10)
    assert np.all(np.isfinite(F))
    assert active[0] is True
    assert active[1] is False
    # With 0.5 norm floor, tacet staff block norm stays strongly attenuated vs active staff
    norm_active = float(np.linalg.norm(F[:, :5]))
    norm_tacet = float(np.linalg.norm(F[:, 5:10]))
    assert norm_tacet < 0.25 * norm_active


def test_extract_single_staff_musicxml_backup_forward_chord_and_attrs() -> None:
    xml_str = """<score-partwise version="3.1">
      <part-list>
        <score-part id="P1"><part-name>Piano</part-name></score-part>
      </part-list>
      <part id="P1">
        <measure number="1">
          <attributes>
            <divisions>4</divisions>
            <staves>2</staves>
            <time number="1"><beats>4</beats><beat-type>4</beat-type></time>
            <time number="2"><beats>4</beats><beat-type>4</beat-type></time>
            <clef><sign>G</sign><line>2</line></clef>
            <clef number="2"><sign>F</sign><line>4</line></clef>
            <staff-details number="1"><staff-lines>5</staff-lines></staff-details>
            <staff-details number="2"><staff-lines>5</staff-lines></staff-details>
          </attributes>
          <!-- Staff 1 note at t=0..8, plus cross-staff chord note on staff 2 at t=0 -->
          <note><pitch><step>C</step><octave>5</octave></pitch><duration>8</duration><staff>1</staff></note>
          <note><chord/><pitch><step>E</step><octave>3</octave></pitch><duration>8</duration><staff>2</staff></note>
          <!-- Staff 1 note at t=8..16 -->
          <note><pitch><step>D</step><octave>5</octave></pitch><duration>8</duration><staff>1</staff></note>
          <backup><duration>16</duration></backup>
          <!-- Staff 2 silent for beat 1-2, note at t=8..16 -->
          <forward><duration>8</duration></forward>
          <note><pitch><step>G</step><octave>3</octave></pitch><duration>8</duration><staff>2</staff></note>
        </measure>
      </part>
    </score-partwise>"""
    indexed = mfs.index_musicxml_root(ET.fromstring(xml_str))
    s2 = mfs.extract_single_staff_musicxml(indexed, staff_idx=1)
    assert s2 is not None
    out = ET.fromstring(mfs.slice_musicxml_measures(s2, 0, 1))
    m1 = out.find("./part/measure")
    assert m1 is not None

    # Clef without number defaulted to "1" (dropped for staff 2); bass clef kept & renumbered to "1"
    clefs = m1.findall("./attributes/clef")
    assert len(clefs) == 1
    assert clefs[0].findtext("sign") == "F"
    assert clefs[0].attrib.get("number") == "1"
    assert len(m1.findall("./attributes/staff-details")) == 1
    assert len(m1.findall("./attributes/time")) == 1

    # Surviving notes: E3 (at t=0, <chord/> stripped!) and G3 (at t=8, zero redundant backup/forward!)
    notes = m1.findall("./note")
    assert [n.findtext("pitch/step") for n in notes] == ["E", "G"]
    assert notes[0].find("chord") is None
    assert m1.findall("./backup") == []
    assert m1.findall("./forward") == []


def test_tier1_insufficient_decoys_fail_closed_and_repeat_safe() -> None:
    n_staves = 2
    bar_w = 80
    staff_h = 40

    def _build_vis(bank_ink: np.ndarray, ranges: list[tuple[int, int]], zero_staff1: bool = False):
        sys_h = n_staves * staff_h
        page_w = 600
        page_h = len(ranges) * (sys_h + 40) + 40
        deskewed = np.full((page_h, page_w), 240, dtype=np.uint8)
        ink_vert = np.zeros((page_h, page_w), dtype=bool)
        sys_crops = []
        for s_i, (m0, m1) in enumerate(ranges):
            sub = bank_ink[:, m0 * bar_w : m1 * bar_w].copy()
            if zero_staff1:
                sub[staff_h:, :] = 0.0
            content_w = 460
            n_bars = m1 - m0
            pre_frac = float(np.clip(0.32 / max(2, n_bars), 0.04, 0.11))
            pre_w = int(round(content_w * pre_frac / (1.0 - pre_frac)))
            sub_rs = cv2.resize(sub, (content_w, sys_h), interpolation=cv2.INTER_AREA)
            y0 = 20 + s_i * (sys_h + 40)
            x0 = 40
            # Draw clef-like ink in left preamble so manuscript trimmer preserves preamble width
            deskewed[y0 + 10 : y0 + 26, x0 + 2 : x0 + 8] = 40
            deskewed[y0 + staff_h + 10 : y0 + staff_h + 26, x0 + 2 : x0 + 8] = 40
            deskewed[y0 : y0 + sys_h, x0 + pre_w : x0 + pre_w + content_w] = np.clip(
                240.0 - 200.0 * sub_rs, 20, 240
            ).astype(np.uint8)
            sys_crops.append({
                "y_top": y0,
                "y_bot": y0 + sys_h,
                "x_left": x0,
                "x_right": x0 + pre_w + content_w,
                "staves": [
                    {"y_top": y0 + 6, "y_bot": y0 + staff_h - 6, "x_left": x0, "x_right": x0 + pre_w + content_w},
                    {"y_top": y0 + staff_h + 6, "y_bot": y0 + 2 * staff_h - 6, "x_left": x0, "x_right": x0 + pre_w + content_w},
                ],
            })
        return {"deskewed": deskewed, "ink_vert": ink_vert, "system_crops": sys_crops}

    # Case 1: Short 8-measure piece with [0, 4) -> only 1 non-overlapping decoy + rev = 2 < 5 -> FAIL CLOSED!
    short_bank, bands = _make_distinct_measure_bank(8, n_staves, bar_w, staff_h, seed=55)
    short_idx = mfs.index_musicxml_root(_make_sample_musicxml(n_measures=8, two_staves=True))
    vis_short = _build_vis(short_bank, [(0, 4)])
    res_short = mfs.slice_page_modelfree(
        vis_short,
        short_idx,
        n_staves=2,
        render_fn=lambda _i, m0, m1, _ns: (short_bank[:, m0 * bar_w : m1 * bar_w], bands, False),
    )
    assert res_short["accepted"] is False
    assert "tier1_insufficient_decoys_s0" in res_short["reason"]

    # Case 2: Piece with a literal repeat ([16, 20) == [0, 4)) still ACCEPTED via 2nd-highest decoy!
    long_bank, bands = _make_distinct_measure_bank(28, n_staves, bar_w, staff_h, seed=99)
    long_bank[:, 16 * bar_w : 20 * bar_w] = long_bank[:, 0 : 4 * bar_w]
    long_idx = mfs.index_musicxml_root(_make_sample_musicxml(n_measures=28, two_staves=True))
    vis_rep = _build_vis(long_bank, [(0, 4), (4, 8)])
    res_rep = mfs.slice_page_modelfree(
        vis_rep,
        long_idx,
        n_staves=2,
        render_fn=lambda _i, m0, m1, _ns: (long_bank[:, m0 * bar_w : m1 * bar_w], bands, False),
    )
    assert res_rep["accepted"] is True, f"Literal repeat should pass via 2nd-highest decoy, got {res_rep['reason']}"

    # Case 3: Tier 2 one-sided tacet mismatch (manuscript staff 1 blank, Verovio staff 1 active) -> REJECTED!
    vis_tacet_mismatch = _build_vis(long_bank, [(0, 4), (4, 8)], zero_staff1=True)
    res_tm = mfs.slice_page_modelfree(
        vis_tacet_mismatch,
        long_idx,
        n_staves=2,
        render_fn=lambda _i, m0, m1, _ns: (long_bank[:, m0 * bar_w : m1 * bar_w], bands, False),
    )
    assert res_tm["accepted"] is False
    assert "tier2_" in res_tm["reason"] or "tier3_" in res_tm["reason"]


def test_part_measure_mismatch_fail_closed() -> None:
    indexed = mfs.index_musicxml_root(_make_sample_musicxml(n_measures=6, part2_measures=4))
    assert indexed.part_measure_mismatch is True
    vis = {
        "deskewed": np.full((100, 200), 240, dtype=np.uint8),
        "ink_vert": np.zeros((100, 200), dtype=bool),
        "system_crops": [{"y_top": 0, "y_bot": 100, "x_left": 0, "x_right": 200, "staves": [{"y_top": 20, "y_bot": 80, "x_left": 0, "x_right": 200}]}],
    }
    res = mfs.slice_page_modelfree(vis, indexed, n_staves=1)
    assert res["accepted"] is False
    assert "rej_part_measure_mismatch" in res["reason"]


def test_next_canvas_image_path_and_svg_origin_classes() -> None:
    assert mfs.next_canvas_image_path(Path("/a/rism_100_c01.jpg")) == Path("/a/rism_100_c02.jpg")
    assert mfs.next_canvas_image_path(Path("/a/rism_100_c09.jpg")) == Path("/a/rism_100_c10.jpg")
    assert mfs.next_canvas_image_path(Path("/a/rism_100_c1.png")) == Path("/a/rism_100_c2.png")
    assert mfs.next_canvas_image_path(Path("/a/rism_100_page1.jpg")) is None

    # Multi-token CSS classes, nested <g class="staff">, and non-zero viewBox origin
    svg = """<svg viewBox="0 0 200 100" xmlns="http://www.w3.org/2000/svg">
      <svg class="foo definition-scale bar" viewBox="20 10 2000 1000">
        <g class="page-margin extra" transform="translate(50, 40)">
          <g class="measure m-1">
            <g class="wrapper">
              <g class="staff s-1">
                <path d="M100 100 L900 100"/>
                <path d="M100 140 L900 140"/>
                <path d="M100 180 L900 180"/>
              </g>
            </g>
          </g>
        </g>
      </svg>
    </svg>"""
    parsed = mfs._parse_and_strip_verovio_svg(svg, n_staves=1)
    assert parsed is not None
    _, w_def, h_def, tx_eff, ty_eff, m_xs, s_ys = parsed
    assert (w_def, h_def, tx_eff, ty_eff) == (2000.0, 1000.0, 30.0, 30.0)
    assert m_xs == [(100.0, 900.0)]
    assert s_ys == [(100.0, 180.0)]


def test_resume_preserves_status_counts(tmp_path: Path) -> None:
    import json

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    gold_jsonl = tmp_path / "gold.jsonl"
    fake_aser = tmp_path / "aser.exe"
    fake_aser.write_bytes(b"")

    # Pre-populate state.jsonl with 2 previously processed rows
    img1 = tmp_path / "p1.jpg"
    img2 = tmp_path / "p2.jpg"
    stem1 = mfs.sanitize_page_stem("1001", img1)
    stem2 = mfs.sanitize_page_stem("1002", img2)
    gold_rows = [
        {"rism_id": "1001", "image_path": str(img1), "work_id": "w1", "work_source_path": "", "work_n_staves": 1},
        {"rism_id": "1002", "image_path": str(img2), "work_id": "w2", "work_source_path": "", "work_n_staves": 1},
    ]
    gold_jsonl.write_text("\n".join(json.dumps(r) for r in gold_rows) + "\n", encoding="utf-8")
    state_rows = [
        {"status": "accepted", "page_stem": stem1, "rism_id": "1001", "entry": {"rism_id": "1001", "page_stem": stem1}},
        {"status": "rej_missing_img", "page_stem": stem2, "rism_id": "1002", "reason": "missing_image"},
    ]
    (out_dir / "state.jsonl").write_text("\n".join(json.dumps(r) for r in state_rows) + "\n", encoding="utf-8")

    mfs.main([
        "--gold-jsonl", str(gold_jsonl),
        "--out-dir", str(out_dir),
        "--aser-bin", str(fake_aser),
        "--resume",
    ])
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status_counts"]["accepted"] == 1
    assert summary["status_counts"]["rej_missing_img"] == 1
    assert summary["final_accepted"] == 1
    assert summary["resumed_skipped"] == 2
