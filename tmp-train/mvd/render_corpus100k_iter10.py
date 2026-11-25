#!/usr/bin/env python3
"""Stage 3: 100k Quota-Balanced Multi-Renderer Render Worker for UniOMR Iter 10.

Strictly CPU-only (CUDA_VISIBLE_DEVICES=""), max 4 parallel workers (RAM guard).
Reads `D:\\tmp\\corpus100k\\corpus100k_manifest.jsonl`, performs rare-bucket-first
stratified selection across all permissive `.mxl`, `.musicxml`, and `.krn` sources:
  - Jazz / Lead Sheets (`openewld_jazz` + PDMX jazz)
  - Drums / Percussion (`v2_percussion_kit` -> `clef-p-2`)
  - Full Conductor / Orchestral & Big Band & Organ 3-staff
  - Alto (`C-3`), Tenor (`C-4`), Octave (`8vb`/`8va`), Historical C-clefs,
    Mid-bar clef changes, Transposing multi-key
  - Odd/Asymmetric meters, Polymeter, Complex tuplets, Double accidentals
  - NIFC Polish & Chopin `.krn` (via music21 -> MusicXML -> Verovio/OSMD + assembly-serialize)
  - OpenScore Lieder / String Quartets / MalerLab OSSQ
  - Multi-page scores: explicitly renders interior pages (`p.2..N-1`) AND trailing
    final stub pages (`p.N`), not just page 1!
  - Wide bar-count-per-line (`1..8` + irregular) & spacing-stress variation across
    Verovio (all 5 SMuFL fonts: Leipzig, Bravura, Petaluma, Gootville, Leland)
    and OSMD/VexFlow (3 fonts x 3 styles).

Exact QA guarantees:
  - 0 eval leakage (`suite31_hw`, `gate20`, `mozart-k458`, NC)
  - Exact rendered measure-range parity per `<sys>` (`sum(svg_m) == xml_m`)
  - 0 OOV against `data.VOCAB`, 100% `is_stream_canonical_i1`, 100% `PrefixState(strict=False)`
  - Resumable (`--resume`) and deterministic (`--seed`).
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from PIL import Image

MVD_DIR = Path(__file__).resolve().parent
REPO_ROOT = MVD_DIR.parent.parent
if str(MVD_DIR) not in sys.path:
    sys.path.insert(0, str(MVD_DIR))

import data as mvd_data  # noqa: E402
import multi_engrave  # noqa: E402
import recanon_i1  # noqa: E402
import render_osmd_dataset as rosmd  # noqa: E402
import render_verovio_dataset as rvd  # noqa: E402

VOCAB_SET = frozenset(mvd_data.VOCAB)
MAX_PAGE_TOKENS = 3800
MAX_SCORE_PAGES_VERIFY = 24
MAX_KRN_BYTES = 180_000

# Rare-bucket priority order & target score pool sizes (round-robin interleaved)
BUCKET_QUOTAS: Tuple[Tuple[str, int], ...] = (
    ("jazz_openewld", 500),
    ("jazz_pdmx", 350),
    ("percussion_kit", 500),
    ("conductor_orchestral_full", 400),
    ("big_band_jazz_combo", 350),
    ("organ_3staff", 220),
    ("clef_alto_c3", 320),
    ("clef_tenor_c4", 320),
    ("clef_octave", 320),
    ("clef_historical_c", 320),
    ("clef_mid_measure", 320),
    ("clef_transposing", 320),
    ("meter_asymmetric_odd", 320),
    ("meter_polymeter", 320),
    ("meter_complex_tuplets", 320),
    ("meter_swing_syncopation", 260),
    ("key_double_accidentals", 320),
    ("key_mid_system_changes", 260),
    ("nifc_chopin_krn", 360),
    ("nifc_polish_krn", 500),
    ("openscore_lieder_mxl", 350),
    ("openscore_sq_ossq", 180),
    ("general_multipage", 800),
)


def _safe_slug(s: str, max_len: int = 44) -> str:
    h = hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:10]
    clean = re.sub(r"[^a-zA-Z0-9_-]+", "_", s).strip("_")
    if len(clean) > max_len:
        clean = clean[:max_len].rstrip("_")
    return f"{clean}_{h}" if clean else h


def _classify_rare_buckets(rec: Dict[str, Any]) -> List[str]:
    """Return ordered list of rare buckets this score belongs to (rarest first)."""
    d = rec["dims"]
    t = rec["v2_tags"]
    src = rec["source_id"]
    out: List[str] = []

    if src == "openewld_jazz":
        out.append("jazz_openewld")
    elif d.get("01_genre_tradition") == "jazz_lead_sheets":
        out.append("jazz_pdmx")

    if "v2_percussion_kit" in t:
        out.append("percussion_kit")
    if d.get("02_ensemble_score_type") == "organ_3staff":
        out.append("organ_3staff")
    if d.get("02_ensemble_score_type") == "big_band_jazz_combo":
        out.append("big_band_jazz_combo")
    if d.get("02_ensemble_score_type") == "conductor_orchestral_full":
        out.append("conductor_orchestral_full")

    c = d.get("03_clefs_transposition", "")
    clef_map = {
        "alto_c3": "clef_alto_c3",
        "tenor_c4": "clef_tenor_c4",
        "octave_clefs": "clef_octave",
        "historical_c_clefs": "clef_historical_c",
        "mid_measure_clef_changes": "clef_mid_measure",
        "transposing_multi_key": "clef_transposing",
    }
    if c in clef_map:
        out.append(clef_map[c])

    m = d.get("04_meter_rhythm_complexity", "")
    meter_map = {
        "asymmetric_odd_meters": "meter_asymmetric_odd",
        "polymeter_meter_changes": "meter_polymeter",
        "complex_tuplets": "meter_complex_tuplets",
        "swing_syncopation_crossbeam": "meter_swing_syncopation",
    }
    if m in meter_map:
        out.append(meter_map[m])

    k = d.get("05_keys_accidentals_tonality", "")
    if k == "double_accidentals":
        out.append("key_double_accidentals")
    elif k == "mid_system_key_changes":
        out.append("key_mid_system_changes")

    if src == "nifc_chopin_first_editions":
        out.append("nifc_chopin_krn")
    elif src == "nifc_polish_scores":
        out.append("nifc_polish_krn")
    elif src == "openscore_lieder":
        out.append("openscore_lieder_mxl")
    elif src in ("openscore_string_quartets", "malerlab_ossq_omr"):
        out.append("openscore_sq_ossq")

    if not out or rec.get("total_pages", 1) >= 3:
        out.append("general_multipage")
    return out


def load_and_select_candidates(
    manifest_path: Path,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Load corpus100k_manifest.jsonl, enforce strict eval filter, and select rare-first pool."""
    policy, blocked = rvd.build_strict_eval_filter(REPO_ROOT)
    stats = collections.Counter()
    scores: Dict[str, Dict[str, Any]] = {}

    with manifest_path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            fmt = r.get("format", "")
            if fmt not in ("mxl", "musicxml", "krn"):
                continue
            sid = r["score_id"]
            if sid in scores:
                scores[sid]["v2_tags"].update(r.get("v2_tags") or [])
                continue

            # Strict leakage & NC check on path, score_id, title, composer
            probe_str = f"{r.get('path', '')} {sid} {r.get('title', '')} {r.get('composer', '')}"
            if rvd.is_path_blocked(probe_str, policy, blocked):
                stats["excluded_eval_or_nc"] += 1
                continue

            tp = int(r.get("total_pages") or 1)
            if tp > MAX_SCORE_PAGES_VERIFY:
                stats["excluded_too_long"] += 1
                continue

            path_str = str(r.get("path", ""))
            if "::" not in path_str:
                pp = Path(path_str)
                if not pp.is_file():
                    stats["excluded_missing_file"] += 1
                    continue
                if fmt == "krn" and pp.stat().st_size > MAX_KRN_BYTES:
                    stats["excluded_large_krn"] += 1
                    continue

            scores[sid] = {
                "score_id": sid,
                "source_id": str(r.get("source_id", "")),
                "format": fmt,
                "tier": str(r.get("supervision_tier", "")),
                "path": path_str,
                "total_pages": tp,
                "dims": dict(r.get("dimensions") or {}),
                "v2_tags": set(r.get("v2_tags") or []),
            }

    stats["eligible_unique_scores"] = len(scores)

    # Group into buckets (each score may qualify for multiple buckets; assign uniquely by quota)
    bucket_pools: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for rec in scores.values():
        b_list = _classify_rare_buckets(rec)
        rec["all_buckets"] = b_list
        for b in b_list:
            bucket_pools[b].append(rec)

    rng = random.Random(seed)
    for b in bucket_pools:
        # Prefer multi-page scores slightly within buckets while staying randomized
        rng.shuffle(bucket_pools[b])

    chosen_ids: Set[str] = set()
    by_bucket_selected: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)

    for b_name, quota in BUCKET_QUOTAS:
        pool = bucket_pools.get(b_name, [])
        taken = 0
        for rec in pool:
            if taken >= quota:
                break
            sid = rec["score_id"]
            if sid in chosen_ids:
                continue
            chosen_ids.add(sid)
            item = {
                "score_id": sid,
                "slug": _safe_slug(sid),
                "source_id": rec["source_id"],
                "format": rec["format"],
                "tier": rec["tier"],
                "path": rec["path"],
                "total_pages": rec["total_pages"],
                "primary_bucket": b_name,
                "all_buckets": list(rec["all_buckets"]),
                "dims": rec["dims"],
            }
            by_bucket_selected[b_name].append(item)
            taken += 1
        stats[f"pool_{b_name}"] = taken

    # Round-robin interleave across buckets so early stopping still hits EVERY rare stratum!
    interleaved: List[Dict[str, Any]] = []
    bucket_names = [b for b, _ in BUCKET_QUOTAS if by_bucket_selected.get(b)]
    cursors = {b: 0 for b in bucket_names}
    while True:
        added_any = False
        for b in bucket_names:
            c_idx = cursors[b]
            lst = by_bucket_selected[b]
            if c_idx < len(lst):
                interleaved.append(lst[c_idx])
                cursors[b] = c_idx + 1
                added_any = True
        if not added_any:
            break

    stats["selected_total"] = len(interleaved)
    return interleaved, dict(stats)


def pre_extract_tar_members(cands: Sequence[Dict[str, Any]], cache_dir: Path) -> Dict[str, str]:
    """Single streaming pass over any tar.gz archives referenced via `archive::member`."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    needed_by_tar: Dict[str, Dict[str, Path]] = collections.defaultdict(dict)
    resolved: Dict[str, str] = {}

    for c in cands:
        p = c["path"]
        if "::" not in p:
            resolved[c["score_id"]] = p
            continue
        tar_path, member = p.split("::", 1)
        out_p = cache_dir / f"{c['slug']}.{c['format']}"
        if out_p.is_file() and out_p.stat().st_size > 0:
            resolved[c["score_id"]] = str(out_p)
        else:
            needed_by_tar[tar_path][member] = out_p
            needed_by_tar[tar_path]["./" + member.lstrip("./")] = out_p
            resolved[c["score_id"]] = str(out_p)

    for tar_path, want_map in needed_by_tar.items():
        if not want_map:
            continue
        # Deduplicate target paths count
        unique_targets = set(want_map.values())
        print(
            f"[corpus100k-render] streaming {len(unique_targets)} members from {tar_path} ...",
            flush=True,
        )
        t0 = time.time()
        found = 0
        done_targets: Set[Path] = set()
        with tarfile.open(tar_path, mode="r|gz") as tf:
            for ti in tf:
                if not ti.isfile():
                    continue
                target = want_map.get(ti.name)
                if target is None:
                    norm = ti.name.lstrip("./")
                    target = want_map.get(norm)
                if target is not None and target not in done_targets:
                    src_f = tf.extractfile(ti)
                    if src_f is not None:
                        data = src_f.read()
                        tmp = target.with_suffix(target.suffix + ".tmp")
                        tmp.write_bytes(data)
                        os.replace(tmp, target)
                        done_targets.add(target)
                        found += 1
                        if found >= len(unique_targets):
                            break
        print(
            f"[corpus100k-render] extracted {found}/{len(unique_targets)} members in {time.time()-t0:.1f}s",
            flush=True,
        )

    return resolved


def materialize_musicxml(src_path: Path, fmt: str, work_dir: Path) -> Optional[Path]:
    """Return a clean local `.musicxml` path from `.musicxml`, `.mxl`, or `.krn`."""
    if fmt == "musicxml":
        return src_path if src_path.is_file() else None
    if fmt == "mxl":
        if not src_path.is_file():
            return None
        try:
            with zipfile.ZipFile(src_path) as zf:
                # Prefer container.xml rootfile if present
                xml_name: Optional[str] = None
                if "META-INF/container.xml" in zf.namelist():
                    c_txt = zf.read("META-INF/container.xml").decode("utf-8", errors="ignore")
                    m = re.search(r'full-path="([^"]+)"', c_txt)
                    if m and m.group(1) in zf.namelist():
                        xml_name = m.group(1)
                if xml_name is None:
                    cands = [
                        n
                        for n in zf.namelist()
                        if n.lower().endswith((".musicxml", ".xml"))
                        and not n.upper().startswith("META-INF")
                    ]
                    if not cands:
                        return None
                    xml_name = cands[0]
                out_xml = work_dir / "score.musicxml"
                out_xml.write_bytes(zf.read(xml_name))
                return out_xml
        except Exception:
            return None
    if fmt == "krn":
        if not src_path.is_file():
            return None
        try:
            import music21

            sc = music21.converter.parse(str(src_path), storePickle=False, forceSource=True)
            out_xml = work_dir / "score.musicxml"
            sc.write("musicxml", fp=str(out_xml))
            return out_xml if out_xml.is_file() else None
        except Exception:
            return None
    return None


def pick_pages_with_roles(n_pages: int, max_pages: int, rng: random.Random) -> List[Tuple[int, str]]:
    """Select up to `max_pages` 1-based pages, explicitly covering interior & final stub pages."""
    if n_pages <= 1:
        return [(1, "single")]
    if n_pages <= max_pages:
        out: List[Tuple[int, str]] = []
        for p in range(1, n_pages + 1):
            role = "first" if p == 1 else ("final_stub" if p == n_pages else "interior")
            out.append((p, role))
        return out

    # Multi-page score with n_pages > max_pages:
    # Always prioritize interior pages + final_stub, and include page 1 ~50% of the time.
    chosen: Dict[int, str] = {}
    chosen[n_pages] = "final_stub"
    if rng.random() < 0.50 and len(chosen) < max_pages:
        chosen[1] = "first"
    interior = list(range(2, n_pages))
    rng.shuffle(interior)
    for p in interior:
        if len(chosen) >= max_pages:
            break
        chosen[p] = "interior"
    if len(chosen) < max_pages and 1 not in chosen:
        chosen[1] = "first"
    return sorted(chosen.items())


def process_one_verovio_score(job: Dict[str, Any]) -> Dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import verovio

    try:
        verovio.enableLog(verovio.LOG_ERROR)
    except Exception:
        pass

    score_id = job["score_id"]
    slug = job["slug"]
    src_path = Path(job["local_path"])
    fmt = job["format"]
    out_dir = Path(job["out_dir"])
    font = job["font"]
    preset = job["preset"]
    seed = int(job["seed"])
    max_pages = int(job["max_pages"])
    aser_bin = Path(job["aser_bin"])
    spacing_stress = bool(job["spacing_stress"])

    rng = random.Random(seed)
    t0 = time.time()
    pages_out: List[Dict[str, Any]] = []
    rej = collections.Counter()

    with tempfile.TemporaryDirectory() as work_td:
        work_p = Path(work_td)
        xml_path = materialize_musicxml(src_path, fmt, work_p)
        if xml_path is None:
            return {
                "score_id": score_id,
                "pages": [],
                "rejections": {f"materialize_{fmt}_fail": 1},
                "elapsed_s": round(time.time() - t0, 2),
            }

        total_xml_measures = rvd.count_xml_measures(xml_path)
        if total_xml_measures < 2:
            return {
                "score_id": score_id,
                "pages": [],
                "rejections": {"too_few_measures": 1},
                "elapsed_s": round(time.time() - t0, 2),
            }

        scale = rng.randint(preset["scale_range"][0], preset["scale_range"][1])
        spacing_lin = round(rng.uniform(0.20, 0.35), 3)
        spacing_nonlin = round(rng.uniform(0.50, 0.70), 3)
        margin_tb = rng.randint(40, 85)
        margin_lr = rng.randint(40, 85)
        bar_lw = round(rng.uniform(0.24, 0.38), 2)
        staff_lw = round(rng.uniform(0.12, 0.22), 2)

        stress_meta: Dict[str, Any] = {}
        forced_pattern: Optional[List[int]] = None

        try:
            tk = verovio.toolkit()
            vrv_opts: Dict[str, Any] = {
                "font": font,
                "pageWidth": int(preset["pageWidth"]),
                "pageHeight": int(preset["pageHeight"]),
                "scale": int(scale),
                "spacingLinear": float(spacing_lin),
                "spacingNonLinear": float(spacing_nonlin),
                "pageMarginTop": int(margin_tb),
                "pageMarginBottom": int(margin_tb),
                "pageMarginLeft": int(margin_lr),
                "pageMarginRight": int(margin_lr),
                "barLineWidth": float(bar_lw),
                "staffLineWidth": float(staff_lw),
                "adjustPageHeight": False,
                "breaks": "auto",
                "condense": "none",
                "header": "none",
                "footer": "none",
                "svgAdditionalAttribute": ["measure@n"],
            }

            if spacing_stress:
                if rng.random() < 0.60:
                    k_bars = rng.choice([1, 2, 2, 3, 4, 5, 6, 6, 7, 7, 8, 8])
                    forced_pattern = [k_bars]
                else:
                    irreg_presets = [
                        [2, 6, 3, 5],
                        [1, 4, 2],
                        [3, 7, 2, 5],
                        [1, 6, 2, 4],
                        [5, 2, 7, 3],
                        [2, 8, 3],
                        [4, 1, 6, 2],
                        [6, 2, 5, 1],
                    ]
                    if rng.random() < 0.65:
                        forced_pattern = list(rng.choice(irreg_presets))
                    else:
                        forced_pattern = [rng.randint(1, 8) for _ in range(rng.randint(3, 5))]

                max_pat = max(forced_pattern)
                if max_pat >= 7:
                    scale = rng.randint(28, 40)
                    spacing_lin = round(rng.uniform(0.12, 0.30), 3)
                elif max_pat <= 2:
                    scale = rng.randint(38, 52)
                    spacing_lin = round(rng.uniform(0.28, 0.55), 3)
                else:
                    scale = rng.randint(32, 48)
                    spacing_lin = round(rng.uniform(0.12, 0.55), 3)

                spacing_nonlin = round(rng.uniform(0.35, 0.85), 3)
                even_note_spacing = rng.random() < 0.30
                spacing_system = rng.randint(2, 24)
                spacing_staff = rng.randint(4, 22)
                justify_vert = rng.random() < 0.50
                staff_lw = round(rng.uniform(0.10, 0.24), 2)
                stem_w = round(rng.uniform(0.10, 0.22), 2)
                bar_lw = round(rng.uniform(0.14, 0.32), 2)
                beam_max_slope = rng.randint(5, 20)
                breaks_mode = "encoded" if rng.random() < 0.85 else "smart"

                vrv_opts.update(
                    {
                        "scale": int(scale),
                        "spacingLinear": float(spacing_lin),
                        "spacingNonLinear": float(spacing_nonlin),
                        "evenNoteSpacing": bool(even_note_spacing),
                        "spacingSystem": int(spacing_system),
                        "spacingStaff": int(spacing_staff),
                        "justifyVertically": bool(justify_vert),
                        "staffLineWidth": float(staff_lw),
                        "stemWidth": float(stem_w),
                        "barLineWidth": float(bar_lw),
                        "beamMaxSlope": int(beam_max_slope),
                        "breaks": breaks_mode,
                    }
                )
                stress_meta = {
                    "spacing_stress": True,
                    "target_bars_pattern": forced_pattern,
                    "breaks_mode": breaks_mode,
                }

            tk.setOptions(vrv_opts)
            if forced_pattern is not None:
                xml_str = rvd.inject_system_breaks_xml(xml_path, forced_pattern)
                loaded = tk.loadData(xml_str)
            else:
                loaded = tk.loadFile(str(xml_path))
            if not loaded:
                return {
                    "score_id": score_id,
                    "pages": [],
                    "rejections": {"verovio_load_fail": 1},
                    "elapsed_s": round(time.time() - t0, 2),
                }

            n_pages = tk.getPageCount()
            if n_pages < 1 or n_pages > MAX_SCORE_PAGES_VERIFY:
                return {
                    "score_id": score_id,
                    "pages": [],
                    "rejections": {"pages_out_of_range": 1},
                    "elapsed_s": round(time.time() - t0, 2),
                }

            # Pass 1: Render SVGs for ALL pages to verify exact total measure count parity
            # and track exact starting measure for every page (enables interior & stub pages!)
            all_svgs: List[str] = []
            all_sys_counts: List[List[int]] = []
            page_start_m: List[int] = []
            cur_m = 1
            for p in range(1, n_pages + 1):
                svg = tk.renderToSVG(p)
                sc = rvd.parse_verovio_svg_measure_counts(svg)
                page_start_m.append(cur_m)
                cur_m += sum(sc)
                all_svgs.append(svg)
                all_sys_counts.append(sc)

            total_svg_measures = cur_m - 1
            if total_svg_measures != total_xml_measures:
                return {
                    "score_id": score_id,
                    "pages": [],
                    "rejections": {"measure_count_mismatch": 1},
                    "elapsed_s": round(time.time() - t0, 2),
                }

            # Select up to max_pages explicitly covering interior + final stub pages
            picked = pick_pages_with_roles(n_pages, max_pages, rng)

            pages_dir = out_dir / "pages"
            beks_dir = out_dir / "bekerns"
            pages_dir.mkdir(parents=True, exist_ok=True)
            beks_dir.mkdir(parents=True, exist_ok=True)

            for pi, page_role in picked:
                svg = all_svgs[pi - 1]
                sc = all_sys_counts[pi - 1]
                if not sc:
                    rej["empty_page"] += 1
                    continue
                m_cursor = page_start_m[pi - 1]
                spans: List[Tuple[int, int]] = []
                for nm in sc:
                    spans.append((m_cursor, m_cursor + nm - 1))
                    m_cursor += nm

                page_stem = f"c100k_{slug}__vrv_{font}_{preset['name']}{'_sp' if spacing_stress else ''}_p{pi}"
                sys_bek_raws: List[str] = []
                page_tok_total = 0
                page_ok = True

                bek_td = work_p / f"bek_p{pi}"
                bek_td.mkdir(parents=True, exist_ok=True)
                for si, (sm, em) in enumerate(spans):
                    s_stem = f"s{si}"
                    bek_path, n_tok, err = multi_engrave.emit_and_validate_bekern(
                        xml_path,
                        sm,
                        em,
                        bek_td,
                        s_stem,
                        aser_bin=aser_bin,
                        min_tokens=1,
                        max_tokens=1200,
                    )
                    if bek_path is None:
                        rej[f"bekern_{(err or 'err').split(':', 1)[0]}"] += 1
                        page_ok = False
                        break
                    raw = bek_path.read_text(encoding="utf-8")
                    toks = raw.split()
                    if any(t not in VOCAB_SET for t in toks):
                        rej["oov"] += 1
                        page_ok = False
                        break
                    if not recanon_i1.is_stream_canonical_i1(raw):
                        rej["non_canonical_i1"] += 1
                        page_ok = False
                        break
                    if not rvd.check_prefix_state(toks):
                        rej["prefix_state_fail"] += 1
                        page_ok = False
                        break
                    sys_bek_raws.append(raw)
                    page_tok_total += len(toks) + 1

                if not page_ok:
                    continue
                if page_tok_total > MAX_PAGE_TOKENS:
                    rej["too_many_tokens"] += 1
                    continue

                img = rvd.svg_to_rgb_png(svg, width_px=int(preset["width_px"]))
                if img.width < 200 or img.height < 200:
                    rej["tiny_img"] += 1
                    continue

                png_path = pages_dir / f"{page_stem}.png"
                tmp_png = png_path.with_suffix(".tmp.png")
                img.save(tmp_png, format="PNG")
                os.replace(tmp_png, png_path)

                bek_paths: List[str] = []
                for si, raw in enumerate(sys_bek_raws):
                    bp = beks_dir / f"{page_stem}_s{si}-bekern.txt"
                    tmp_bp = bp.with_suffix(".txt.tmp")
                    tmp_bp.write_text(raw, encoding="utf-8")
                    os.replace(tmp_bp, bp)
                    bek_paths.append(str(bp))

                row_out: Dict[str, Any] = {
                    "image": str(png_path),
                    "bekerns": bek_paths,
                    "domain": "printed",
                    "renderer": "verovio",
                    "font": font,
                    "layout": preset["name"],
                    "scale": scale,
                    "n_systems": len(spans),
                    "measure_spans": spans,
                    "n_tokens": page_tok_total,
                    "width": img.width,
                    "height": img.height,
                    "score_id": score_id,
                    "source_id": job["source_id"],
                    "src_format": fmt,
                    "primary_bucket": job["primary_bucket"],
                    "all_buckets": job["all_buckets"],
                    "page_num": pi,
                    "total_pages": n_pages,
                    "page_role": page_role,
                }
                if stress_meta:
                    row_out.update(stress_meta)
                pages_out.append(row_out)

        except Exception as exc:
            rej[f"exc_{type(exc).__name__}"] += 1

    return {
        "score_id": score_id,
        "pages": pages_out,
        "rejections": dict(rej),
        "elapsed_s": round(time.time() - t0, 2),
    }


def run_osmd_phase(
    cands: Sequence[Dict[str, Any]],
    resolved_paths: Dict[str, str],
    out_dir: Path,
    aser_bin: Path,
    target_osmd_pages: int,
    seed: int,
    resume: bool,
) -> List[Dict[str, Any]]:
    """Render short rare-bucket scores (<= 32 bars) via OSMD/VexFlow (3 fonts x 3 styles)."""
    if target_osmd_pages <= 0:
        return []
    node_script = Path(r"D:\tmp\osmd-runner\render_osmd_batch.mjs")
    if not node_script.is_file():
        print(f"[corpus100k-osmd] WARN: missing {node_script}, skipping OSMD phase", flush=True)
        return []

    pages_dir = out_dir / "pages"
    beks_dir = out_dir / "bekerns"
    pages_dir.mkdir(parents=True, exist_ok=True)
    beks_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "_osmd_resume_state.jsonl"

    done_ids: Set[str] = set()
    valid_pages: List[Dict[str, Any]] = []
    if resume and state_path.is_file():
        for ln in state_path.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            try:
                rec = json.loads(ln)
                done_ids.add(rec["score_id"])
                if (
                    rec.get("valid")
                    and Path(rec["row"]["image"]).is_file()
                    and all(Path(b).is_file() for b in rec["row"]["bekerns"])
                ):
                    valid_pages.append(rec["row"])
            except Exception:
                continue
        print(
            f"[corpus100k-osmd] resumed: {len(done_ids)} attempted, {len(valid_pages)} valid",
            flush=True,
        )
    elif state_path.exists():
        state_path.unlink()

    if len(valid_pages) >= target_osmd_pages:
        return valid_pages

    # Pick short scores (total_pages <= 2) from mxl/musicxml (fast to materialize) across all rare buckets
    short_cands = [
        c
        for c in cands
        if c["format"] in ("mxl", "musicxml")
        and c["total_pages"] <= 2
        and c["score_id"] not in done_ids
        and Path(resolved_paths.get(c["score_id"], "")).is_file()
    ]
    print(
        f"[corpus100k-osmd] starting OSMD phase: {len(short_cands)} short rare candidates, target={target_osmd_pages}",
        flush=True,
    )

    t0 = time.time()
    rej = collections.Counter()
    batch_size = 120
    idx = 0

    with state_path.open("a", encoding="utf-8") as sf:
        while len(valid_pages) < target_osmd_pages and idx < len(short_cands):
            chunk = short_cands[idx : idx + batch_size]
            idx += len(chunk)

            with tempfile.TemporaryDirectory() as td:
                td_p = Path(td)
                node_jobs: List[Dict[str, Any]] = []
                meta_by_sid: Dict[str, Dict[str, Any]] = {}

                for j_i, c in enumerate(chunk):
                    sid = c["score_id"]
                    src_p = Path(resolved_paths[sid])
                    sub_td = td_p / f"m_{j_i}"
                    sub_td.mkdir(parents=True, exist_ok=True)
                    xp = materialize_musicxml(src_p, c["format"], sub_td)
                    if xp is None:
                        rej["materialize_fail"] += 1
                        sf.write(json.dumps({"score_id": sid, "valid": False, "error": "materialize_fail"}) + "\n")
                        continue
                    nm = rvd.count_xml_measures(xp)
                    if not (4 <= nm <= 32):
                        rej["measures_range"] += 1
                        sf.write(json.dumps({"score_id": sid, "valid": False, "error": "measures_range"}) + "\n")
                        continue

                    g_idx = len(done_ids) + idx - len(chunk) + j_i
                    font = rosmd.OSMD_FONTS[g_idx % len(rosmd.OSMD_FONTS)]
                    dp = rosmd.OSMD_DRAWING_PARAMS[(g_idx // len(rosmd.OSMD_FONTS)) % len(rosmd.OSMD_DRAWING_PARAMS)]
                    vw = rosmd.OSMD_VIEWPORTS[
                        (g_idx // (len(rosmd.OSMD_FONTS) * len(rosmd.OSMD_DRAWING_PARAMS)))
                        % len(rosmd.OSMD_VIEWPORTS)
                    ]
                    jseed = int(hashlib.sha256(f"{seed}:osmd:{sid}".encode()).hexdigest()[:8], 16)
                    rng = random.Random(jseed)
                    spacing_stress = rng.random() < 0.50
                    zoom = round(rng.uniform(0.78, 1.10), 2)
                    margin_px = rng.randint(20, 52)
                    draw_title = rng.random() < 0.20
                    max_h = 2800 if spacing_stress else 2400
                    xml_for_node = str(xp)
                    new_sys_from_xml = False
                    engraving_rules: Dict[str, Any] = {}
                    stress_info: Dict[str, Any] = {}

                    if spacing_stress:
                        if rng.random() < 0.60:
                            mps = rng.choice([2, 2, 3, 4, 5, 6, 6, 7, 7, 8, 8])
                            engraving_rules["RenderXMeasuresPerLineAkaSystem"] = mps
                            target_pat = [mps]
                        else:
                            irreg_presets = [
                                [2, 6, 3, 5],
                                [1, 4, 2],
                                [3, 7, 2],
                                [2, 5, 1, 4],
                                [6, 2, 4],
                                [1, 5, 3, 6],
                            ]
                            target_pat = list(rng.choice(irreg_presets))
                            injected = rvd.inject_system_breaks_xml(xp, target_pat)
                            tmp_xml = sub_td / "inj.musicxml"
                            tmp_xml.write_text(injected, encoding="utf-8")
                            xml_for_node = str(tmp_xml)
                            new_sys_from_xml = True
                            engraving_rules["RenderXMeasuresPerLineAkaSystem"] = 0

                        max_pat = max(target_pat)
                        if max_pat >= 7:
                            vw = rng.choice([1200, 1350])
                            zoom = round(rng.uniform(0.68, 0.90), 2)
                        elif max_pat <= 2:
                            zoom = round(rng.uniform(0.85, 1.15), 2)
                        else:
                            zoom = round(rng.uniform(0.74, 1.08), 2)

                        min_sky_sys = round(rng.uniform(1.0, 8.0), 2)
                        engraving_rules.update(
                            {
                                "VoiceSpacingMultiplierVexflow": round(rng.uniform(0.65, 1.45), 2),
                                "VoiceSpacingAddendVexflow": round(rng.uniform(1.5, 6.0), 2),
                                "MinSkyBottomDistBetweenSystems": min_sky_sys,
                                "MinimumDistanceBetweenSystems": round(
                                    max(1.5, min_sky_sys + rng.uniform(0.5, 3.0)), 2
                                ),
                                "MinSkyBottomDistBetweenStaves": round(rng.uniform(1.0, 6.0), 2),
                                "StaffDistance": round(rng.uniform(4.0, 12.0), 2),
                                "BetweenStaffDistance": round(rng.uniform(3.0, 10.0), 2),
                                "MeasureLeftMargin": round(rng.uniform(0.2, 2.2), 2),
                                "MeasureRightMargin": round(rng.uniform(0.0, 1.5), 2),
                                "StaffLineWidth": round(rng.uniform(0.08, 0.20), 2),
                                "StretchLastSystemLine": rng.random() < 0.40,
                            }
                        )
                        stress_info = {
                            "spacing_stress": True,
                            "target_bars_pattern": target_pat,
                        }

                    stem = f"c100k_{c['slug']}__osmd_{font}_{dp}{'_sp' if spacing_stress else ''}"
                    node_jobs.append(
                        {
                            "score_id": sid,
                            "xml_path": xml_for_node,
                            "png_path": str(pages_dir / f"{stem}.png"),
                            "drawing_params": dp,
                            "font": font,
                            "viewport_w": vw,
                            "zoom": zoom,
                            "margin_px": margin_px,
                            "draw_title": draw_title,
                            "max_height_px": max_h,
                            "new_system_from_xml": new_sys_from_xml,
                            "engraving_rules": engraving_rules if spacing_stress else None,
                        }
                    )
                    meta_by_sid[sid] = {
                        "cand": c,
                        "xml_path": xp,
                        "xml_measures": nm,
                        "stress_info": stress_info,
                    }

                if not node_jobs:
                    continue

                jobs_json = td_p / "jobs.json"
                res_jsonl = td_p / "results.jsonl"
                jobs_json.write_text(json.dumps(node_jobs), encoding="utf-8")

                proc = subprocess.run(
                    ["node", str(node_script), str(jobs_json), str(res_jsonl)],
                    capture_output=True,
                    text=True,
                    timeout=900,
                    check=False,
                )
                if proc.returncode != 0:
                    print(
                        f"[corpus100k-osmd] WARN node rc={proc.returncode}: {(proc.stderr or '')[-240:]}",
                        flush=True,
                    )
                if not res_jsonl.is_file():
                    continue

                for ln in res_jsonl.read_text(encoding="utf-8").splitlines():
                    if not ln.strip():
                        continue
                    r = json.loads(ln)
                    sid = r["score_id"]
                    meta = meta_by_sid[sid]
                    c = meta["cand"]
                    if not r.get("ok"):
                        rej[f"node_{(r.get('error') or 'err').split(':', 1)[0]}"] += 1
                        sf.write(json.dumps({"score_id": sid, "valid": False, "error": r.get("error")}) + "\n")
                        continue

                    png_path = Path(r["png_path"])
                    if int(r["nMeasures"]) != int(meta["xml_measures"]):
                        rej["measure_count_mismatch"] += 1
                        png_path.unlink(missing_ok=True)
                        sf.write(json.dumps({"score_id": sid, "valid": False, "error": "measure_mismatch"}) + "\n")
                        continue

                    spans = [(int(a), int(b)) for a, b in r["spans"]]
                    xp = meta["xml_path"]
                    stem = png_path.stem
                    sys_raws: List[str] = []
                    tok_total = 0
                    page_ok = True
                    err_reason = ""

                    btd_p = td_p / f"bek_{c['slug']}"
                    btd_p.mkdir(parents=True, exist_ok=True)
                    for si, (sm, em) in enumerate(spans):
                        bek_path, n_tok, err = multi_engrave.emit_and_validate_bekern(
                            xp,
                            sm,
                            em,
                            btd_p,
                            f"s{si}",
                            aser_bin=aser_bin,
                            min_tokens=1,
                            max_tokens=1200,
                        )
                        if bek_path is None:
                            err_reason = f"bekern_{(err or 'err').split(':', 1)[0]}"
                            page_ok = False
                            break
                        raw = bek_path.read_text(encoding="utf-8")
                        toks = raw.split()
                        if any(t not in VOCAB_SET for t in toks):
                            err_reason = "oov"
                            page_ok = False
                            break
                        if not recanon_i1.is_stream_canonical_i1(raw):
                            err_reason = "non_canonical_i1"
                            page_ok = False
                            break
                        if not rvd.check_prefix_state(toks):
                            err_reason = "prefix_state_fail"
                            page_ok = False
                            break
                        sys_raws.append(raw)
                        tok_total += len(toks) + 1

                    if not page_ok or tok_total > MAX_PAGE_TOKENS:
                        rej[err_reason or "too_many_tokens"] += 1
                        png_path.unlink(missing_ok=True)
                        sf.write(
                            json.dumps({"score_id": sid, "valid": False, "error": err_reason or "too_many_tokens"})
                            + "\n"
                        )
                        continue

                    try:
                        with Image.open(png_path) as im:
                            im_w, im_h = im.size
                    except Exception:
                        rej["corrupt_png"] += 1
                        png_path.unlink(missing_ok=True)
                        sf.write(json.dumps({"score_id": sid, "valid": False, "error": "corrupt_png"}) + "\n")
                        continue

                    bek_paths: List[str] = []
                    for si, raw in enumerate(sys_raws):
                        bp = beks_dir / f"{stem}_s{si}-bekern.txt"
                        tmp_bp = bp.with_suffix(".txt.tmp")
                        tmp_bp.write_text(raw, encoding="utf-8")
                        os.replace(tmp_bp, bp)
                        bek_paths.append(str(bp))

                    row = {
                        "image": str(png_path),
                        "bekerns": bek_paths,
                        "domain": "printed",
                        "renderer": "osmd",
                        "font": r["font"],
                        "drawing_params": r["drawing_params"],
                        "viewport_w": r["viewport_w"],
                        "zoom": r["zoom"],
                        "n_systems": len(spans),
                        "measure_spans": spans,
                        "n_tokens": tok_total,
                        "width": im_w,
                        "height": im_h,
                        "score_id": sid,
                        "source_id": c["source_id"],
                        "src_format": c["format"],
                        "primary_bucket": c["primary_bucket"],
                        "all_buckets": c["all_buckets"],
                        "page_num": 1,
                        "total_pages": 1,
                        "page_role": "single",
                    }
                    if meta["stress_info"]:
                        row.update(meta["stress_info"])
                    valid_pages.append(row)
                    sf.write(json.dumps({"score_id": sid, "valid": True, "row": row}, separators=(",", ":")) + "\n")
                sf.flush()

            print(
                f"[corpus100k-osmd {time.time()-t0:6.1f}s] valid={len(valid_pages)}/{target_osmd_pages} "
                f"rej={dict(rej.most_common(5))}",
                flush=True,
            )

    return valid_pages


def main() -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--corpus100k-manifest",
        type=Path,
        default=Path(r"D:\tmp\corpus100k\corpus100k_manifest.jsonl"),
    )
    ap.add_argument("--out-dir", type=Path, default=Path(r"D:\tmp\corpus100k\rendered_iter10"))
    ap.add_argument("--target-verovio-pages", type=int, default=6000)
    ap.add_argument("--target-osmd-pages", type=int, default=1000)
    ap.add_argument("--max-pages-per-score", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261201)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    workers = min(4, max(1, int(args.workers)))
    aser_bin = multi_engrave.default_assembly_serialize_bin()
    if not aser_bin.is_file():
        raise FileNotFoundError(f"assembly-serialize not found: {aser_bin}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    vrv_state_path = args.out_dir / "_vrv_resume_state.jsonl"
    manifest_path = args.out_dir / "manifest.jsonl"
    mxl_cache_dir = args.out_dir / "_mxl_cache"

    t_start = time.time()
    cands, sel_stats = load_and_select_candidates(args.corpus100k_manifest, seed=args.seed)
    print(f"[corpus100k-render] selected {len(cands)} rare-balanced scores in {time.time()-t_start:.1f}s", flush=True)
    print(f"  stats={sel_stats}", flush=True)

    # Pre-extract any tar.gz members in one fast streaming pass
    resolved_paths = pre_extract_tar_members(cands, mxl_cache_dir)

    # Phase 1: Verovio multi-font + multi-page (interior + stub) + 50% spacing-stress
    done_scores: Set[str] = set()
    vrv_pages: List[Dict[str, Any]] = []
    if args.resume and vrv_state_path.is_file():
        for ln in vrv_state_path.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            try:
                rec = json.loads(ln)
                done_scores.add(rec["score_id"])
                for pg in rec.get("pages", []):
                    if Path(pg["image"]).is_file() and all(Path(b).is_file() for b in pg["bekerns"]):
                        vrv_pages.append(pg)
            except Exception:
                continue
        print(
            f"[corpus100k-vrv] resumed: {len(done_scores)} scores, {len(vrv_pages)} pages",
            flush=True,
        )
    elif vrv_state_path.exists():
        vrv_state_path.unlink()

    jobs: List[Dict[str, Any]] = []
    for idx, c in enumerate(cands):
        if c["score_id"] in done_scores:
            continue
        lp = resolved_paths.get(c["score_id"], "")
        if not lp or not Path(lp).is_file():
            continue
        font = rvd.VEROVIO_FONTS[idx % len(rvd.VEROVIO_FONTS)]
        preset = rvd.LAYOUT_PRESETS[(idx // len(rvd.VEROVIO_FONTS)) % len(rvd.LAYOUT_PRESETS)]
        spacing_stress = (idx % 2) == 1
        jseed = int(
            hashlib.sha256(f"{args.seed}:vrv:{c['score_id']}:{font}:{int(spacing_stress)}".encode()).hexdigest()[:8],
            16,
        )
        jobs.append(
            {
                "score_id": c["score_id"],
                "slug": c["slug"],
                "source_id": c["source_id"],
                "format": c["format"],
                "primary_bucket": c["primary_bucket"],
                "all_buckets": c["all_buckets"],
                "local_path": lp,
                "out_dir": str(args.out_dir),
                "font": font,
                "preset": preset,
                "seed": jseed,
                "max_pages": args.max_pages_per_score,
                "aser_bin": str(aser_bin),
                "spacing_stress": spacing_stress,
            }
        )

    t0 = time.time()
    rej_tot = collections.Counter()
    ji = 0
    completed = 0

    if len(vrv_pages) < args.target_verovio_pages:
        with vrv_state_path.open("a", encoding="utf-8") as sf:
            while ji < len(jobs) and len(vrv_pages) < args.target_verovio_pages:
                fut_to_job: Dict[concurrent.futures.Future, Dict[str, Any]] = {}
                cur_job: Optional[Dict[str, Any]] = None
                try:
                    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
                        while len(fut_to_job) < workers and ji < len(jobs):
                            j = jobs[ji]
                            fut_to_job[pool.submit(process_one_verovio_score, j)] = j
                            ji += 1

                        while fut_to_job:
                            done_set, _ = concurrent.futures.wait(
                                list(fut_to_job.keys()), return_when=concurrent.futures.FIRST_COMPLETED
                            )
                            for fut in done_set:
                                cur_job = fut_to_job.pop(fut)
                                res = fut.result()
                                cur_job = None
                                sf.write(json.dumps(res, separators=(",", ":")) + "\n")
                                sf.flush()
                                vrv_pages.extend(res.get("pages", []))
                                rej_tot.update(res.get("rejections", {}))
                                completed += 1
                                if completed % 50 == 0 or len(vrv_pages) >= args.target_verovio_pages:
                                    by_font = collections.Counter(p["font"] for p in vrv_pages)
                                    by_role = collections.Counter(p.get("page_role", "?") for p in vrv_pages)
                                    print(
                                        f"[corpus100k-vrv {time.time()-t0:6.1f}s] scores={completed} "
                                        f"pages={len(vrv_pages)}/{args.target_verovio_pages} "
                                        f"roles={dict(by_role)} fonts={dict(by_font)} "
                                        f"rej={dict(rej_tot.most_common(4))}",
                                        flush=True,
                                    )

                            while (
                                len(fut_to_job) < workers
                                and ji < len(jobs)
                                and len(vrv_pages) < args.target_verovio_pages
                            ):
                                j = jobs[ji]
                                fut_to_job[pool.submit(process_one_verovio_score, j)] = j
                                ji += 1
                except concurrent.futures.process.BrokenProcessPool:
                    # Record crashed in-flight scores so we skip them and restart the pool cleanly
                    crashed_ids = [j["score_id"] for j in fut_to_job.values()]
                    if cur_job is not None:
                        crashed_ids.append(cur_job["score_id"])
                    print(
                        f"[corpus100k-vrv] WARN BrokenProcessPool; skipping {len(crashed_ids)} in-flight scores and restarting pool",
                        flush=True,
                    )
                    for csid in crashed_ids:
                        sf.write(
                            json.dumps(
                                {"score_id": csid, "pages": [], "rejections": {"worker_crash": 1}},
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                        rej_tot["worker_crash"] += 1
                    sf.flush()

    # Phase 2: OSMD / VexFlow multi-font + 50% spacing-stress on short rare-bucket scores
    osmd_pages = run_osmd_phase(
        cands=cands,
        resolved_paths=resolved_paths,
        out_dir=args.out_dir,
        aser_bin=aser_bin,
        target_osmd_pages=args.target_osmd_pages,
        seed=args.seed,
        resume=args.resume,
    )

    all_pages = vrv_pages + osmd_pages
    tmp_m = manifest_path.with_suffix(".jsonl.tmp")
    with tmp_m.open("w", encoding="utf-8") as f:
        for pg in all_pages:
            f.write(json.dumps(pg, ensure_ascii=False) + "\n")
    os.replace(tmp_m, manifest_path)

    # Print rich summary
    by_renderer = collections.Counter(p["renderer"] for p in all_pages)
    by_font = collections.Counter(f"{p['renderer']}/{p['font']}" for p in all_pages)
    by_bucket = collections.Counter(p.get("primary_bucket", "unknown") for p in all_pages)
    by_source = collections.Counter(p.get("source_id", "unknown") for p in all_pages)
    by_role = collections.Counter(p.get("page_role", "unknown") for p in all_pages)
    by_stress = collections.Counter(bool(p.get("spacing_stress", False)) for p in all_pages)

    print(f"[corpus100k-render] DONE -> {manifest_path} (total={len(all_pages)} in {time.time()-t_start:.1f}s)", flush=True)
    print(f"  by_renderer={dict(by_renderer)}", flush=True)
    print(f"  by_role={dict(by_role)}", flush=True)
    print(f"  by_spacing_stress={dict(by_stress)}", flush=True)
    print(f"  by_source={dict(by_source)}", flush=True)
    print(f"  by_font={dict(sorted(by_font.items()))}", flush=True)
    print(f"  by_primary_bucket={dict(sorted(by_bucket.items()))}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
