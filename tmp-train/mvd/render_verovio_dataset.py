#!/usr/bin/env python3
"""Render a variation-rich, leak-free Verovio full-page dataset for UniOMR Iter 10.

Strictly CPU-only (CUDA_VISIBLE_DEVICES=""), max 4 parallel workers.
Renders clean permissive MusicXML scores (PDMX + big500) page-by-page across:
  - All 5 built-in Verovio SMuFL fonts: Leipzig, Bravura, Petaluma, Gootville, Leland
  - Diverse page layouts: A4 portrait, Letter portrait, B4 portrait, A4 landscape,
    compact portrait, and wide spacious portrait
  - Varied scale (32..52), linear/non-linear spacing, margins, and line weights

Exact page & <sys> alignment guarantee:
  - Parses <g class="system"> and <g class="measure"> from Verovio's SVG DOM.
  - Enforces sum(svg_measures across all pages) == total_musicxml_measures before
    emitting any page from that score (eliminates multi-measure-rest / pickup drift).
  - Emits per-system bekern+ via `assembly-serialize bekern --measures S..E`,
    validated by `bekern-decode`, 0 OOV against `data.VOCAB`, `PrefixState(strict=False)`,
    and `recanon_i1.is_stream_canonical_i1`.
  - Strict 0-overlap against suite31_hw_manifest.json, gate20/exclusion.json,
    mozart-k458, and non-commercial sources.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import io
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from PIL import Image

MVD_DIR = Path(__file__).resolve().parent
REPO_ROOT = MVD_DIR.parent.parent
if str(MVD_DIR) not in sys.path:
    sys.path.insert(0, str(MVD_DIR))

import build_iter7plus_manifest  # noqa: E402
import build_unlabeled_10k  # noqa: E402
import data as mvd_data  # noqa: E402
import grammar  # noqa: E402
import multi_engrave  # noqa: E402
import recanon_i1  # noqa: E402

VEROVIO_FONTS: Tuple[str, ...] = ("Leipzig", "Bravura", "Petaluma", "Gootville", "Leland")

LAYOUT_PRESETS: Tuple[Dict[str, Any], ...] = (
    {"name": "a4_portrait", "pageWidth": 2100, "pageHeight": 2970, "scale_range": (36, 46), "width_px": 1650},
    {"name": "letter_portrait", "pageWidth": 2159, "pageHeight": 2794, "scale_range": (36, 46), "width_px": 1650},
    {"name": "b4_portrait", "pageWidth": 2500, "pageHeight": 3530, "scale_range": (40, 52), "width_px": 1800},
    {"name": "a4_landscape", "pageWidth": 2970, "pageHeight": 2100, "scale_range": (34, 44), "width_px": 2000},
    {"name": "compact", "pageWidth": 1900, "pageHeight": 2650, "scale_range": (32, 40), "width_px": 1500},
    {"name": "wide_spacious", "pageWidth": 2400, "pageHeight": 3200, "scale_range": (42, 52), "width_px": 1750},
)

VOCAB_SET = frozenset(mvd_data.VOCAB)
MAX_PAGE_TOKENS = 3800


def build_strict_eval_filter(repo_root: Path = REPO_ROOT) -> Tuple[build_unlabeled_10k.ExclusionPolicy, Set[str]]:
    policy = build_unlabeled_10k.ExclusionPolicy.from_repo(repo_root)
    blocked = build_iter7plus_manifest._load_eval_blocklist()
    # Add normalized composer+opus/K tokens from suite31_hw_manifest.json
    s31 = repo_root / "data-gen" / "score-corpus" / "suite31" / "suite31_hw_manifest.json"
    if s31.is_file():
        obj = json.loads(s31.read_text(encoding="utf-8"))
        for p in obj.get("pieces", []):
            slug = str(p.get("slug", "")).strip().lower()
            if slug:
                blocked.add(slug)
                if slug.startswith("hw-"):
                    blocked.add(slug[3:])
                # Also strip trailing movement suffix like -1, -2
                base = re.sub(r"-\d+$", "", slug[3:] if slug.startswith("hw-") else slug)
                if len(base) >= 6:
                    blocked.add(base)
    blocked.add("mozart-k458")
    blocked.add("k458")
    return policy, blocked


_BLOCKED_NORM_CACHE: Dict[int, List[str]] = {}


def is_path_blocked(
    path_str: str,
    policy: build_unlabeled_10k.ExclusionPolicy,
    blocked: Set[str],
) -> bool:
    if policy.is_forbidden_nc(path_str) or policy.is_leaked_eval(path_str):
        return True
    if build_iter7plus_manifest._is_blocked(path_str, blocked):
        return True
    key = id(blocked)
    bn_list = _BLOCKED_NORM_CACHE.get(key)
    if bn_list is None:
        bn_list = [f"-{re.sub(r'[^a-z0-9]+', '-', b)}-" for b in blocked if len(b) >= 6]
        _BLOCKED_NORM_CACHE[key] = bn_list
    norm = f"-{re.sub(r'[^a-z0-9]+', '-', path_str.lower())}-"
    for bn in bn_list:
        if bn in norm:
            return True
    return False


def check_prefix_state(tokens: Sequence[str]) -> bool:
    st = grammar.PrefixState(strict=False)
    for t in tokens:
        tid = mvd_data.TOKEN_TO_ID.get(t)
        if tid is None:
            return False
        allowed_ids, _ = st.allowed()
        if tid not in allowed_ids:
            return False
        try:
            st.step(t)
        except Exception:
            return False
        if st.broken:
            return False
    return True


def count_xml_measures(xml_path: Path) -> int:
    try:
        root = ET.parse(str(xml_path)).getroot()
        part = root.find("part")
        if part is None:
            for el in root.iter():
                if el.tag.endswith("part"):
                    part = el
                    break
        if part is None:
            return 0
        return sum(1 for ch in part if ch.tag.endswith("measure"))
    except Exception:
        return 0


def inject_system_breaks_xml(xml_path: Path, pattern: Sequence[int]) -> str:
    """Return in-memory MusicXML string with forced <print new-system="yes"/> per pattern.

    Strips existing new-system / new-page attributes first so the injected bar-count
    schedule controls line breaks cleanly. Does not modify xml_path on disk.
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    pat = [max(1, int(x)) for x in pattern if int(x) >= 1]
    if not pat:
        return ET.tostring(root, encoding="unicode")

    parts = [el for el in root.iter() if el.tag.endswith("part") and not el.tag.endswith("score-part")]
    for part in parts:
        measures = [ch for ch in part if ch.tag.endswith("measure")]
        n_m = len(measures)
        if n_m < 2:
            continue
        break_at: Set[int] = set()
        pos = 1
        k = 0
        while True:
            pos += pat[k % len(pat)]
            k += 1
            if pos > n_m:
                break
            break_at.add(pos)

        for idx, meas in enumerate(measures, start=1):
            print_el = None
            for ch in meas:
                if ch.tag.endswith("print"):
                    print_el = ch
                    ch.attrib.pop("new-system", None)
                    ch.attrib.pop("new-page", None)
            if idx in break_at:
                if print_el is not None:
                    print_el.set("new-system", "yes")
                else:
                    tag_prefix = ""
                    if meas.tag.startswith("{"):
                        tag_prefix = meas.tag.split("}", 1)[0] + "}"
                    new_p = ET.Element(f"{tag_prefix}print", {"new-system": "yes"})
                    meas.insert(0, new_p)

    return ET.tostring(root, encoding="unicode")


def parse_verovio_svg_measure_counts(svg_text: str) -> List[int]:
    """Return list of measure counts per <g class='system'> on this SVG page."""
    root = ET.fromstring(svg_text)
    sys_counts: List[int] = []
    for el in root.iter():
        if el.tag.endswith("g") and "system" in el.attrib.get("class", "").split():
            n_meas = sum(
                1
                for sub in el.iter()
                if sub.tag.endswith("g") and "measure" in sub.attrib.get("class", "").split()
            )
            if n_meas > 0:
                sys_counts.append(n_meas)
    return sys_counts


def svg_to_rgb_png(svg_text: str, width_px: int) -> Image.Image:
    import resvg_py

    png_bytes = resvg_py.svg_to_bytes(svg_string=svg_text, width=int(width_px))
    rgba = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    bg = Image.new("RGB", rgba.size, (255, 255, 255))
    bg.paste(rgba, mask=rgba.split()[3])
    return bg


def select_candidates(
    pdmx_dir: Path,
    stratum_tags_path: Path,
    big500_dir: Path,
    policy: build_unlabeled_10k.ExclusionPolicy,
    blocked: Set[str],
    seed: int = 20261128,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    rng = random.Random(seed)
    cands: List[Dict[str, Any]] = []
    stats = collections.Counter()

    tags: Dict[str, Any] = {}
    if stratum_tags_path.is_file():
        tags = json.loads(stratum_tags_path.read_text(encoding="utf-8"))

    if pdmx_dir.is_dir():
        for xml_path in sorted(pdmx_dir.glob("*.musicxml")):
            if is_path_blocked(str(xml_path), policy, blocked):
                stats["excluded_eval_or_nc"] += 1
                continue
            stem = xml_path.stem
            meta = tags.get(stem, {})
            sig = meta.get("signals") or {}
            ns = int(sig.get("n_staves", 0) or 0)
            pg = int(sig.get("pages", 1) or 1)
            if ns > 6 or pg > 8:
                stats["excluded_size"] += 1
                continue
            cands.append({
                "score_id": f"pdmx_{stem}",
                "xml_path": str(xml_path),
                "stratum": str(meta.get("stratum", "pdmx_other")),
                "est_pages": pg,
            })

    if big500_dir.is_dir():
        for xml_path in sorted(big500_dir.glob("*.musicxml")):
            if is_path_blocked(str(xml_path), policy, blocked):
                stats["excluded_eval_or_nc"] += 1
                continue
            cands.append({
                "score_id": f"big500_{xml_path.stem}",
                "xml_path": str(xml_path),
                "stratum": "big500",
                "est_pages": 2,
            })

    rng.shuffle(cands)
    stats["total_candidates"] = len(cands)
    return cands, dict(stats)


def process_one_score(job: Dict[str, Any]) -> Dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import verovio
    try:
        verovio.enableLog(verovio.LOG_ERROR)
    except Exception:
        pass

    score_id = job["score_id"]
    xml_path = Path(job["xml_path"])
    out_dir = Path(job["out_dir"])
    font = job["font"]
    preset = job["preset"]
    seed = int(job["seed"])
    max_pages = int(job["max_pages"])
    aser_bin = Path(job["aser_bin"])
    spacing_stress = bool(job.get("spacing_stress", False))

    rng = random.Random(seed)
    scale = rng.randint(preset["scale_range"][0], preset["scale_range"][1])
    spacing_lin = round(rng.uniform(0.20, 0.35), 3)
    spacing_nonlin = round(rng.uniform(0.50, 0.70), 3)
    margin_tb = rng.randint(40, 85)
    margin_lr = rng.randint(40, 85)
    bar_lw = round(rng.uniform(0.24, 0.38), 2)
    staff_lw = round(rng.uniform(0.12, 0.22), 2)

    stress_meta: Dict[str, Any] = {}
    forced_pattern: Optional[List[int]] = None
    breaks_mode = "auto"

    t0 = time.time()
    pages_out: List[Dict[str, Any]] = []
    rej = collections.Counter()

    total_xml_measures = count_xml_measures(xml_path)
    if total_xml_measures < 2:
        return {"score_id": score_id, "pages": [], "rejections": {"too_few_measures": 1}, "elapsed_s": 0.0}

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
            # 60% uniform (weighted toward extremes 1..2 and 6..8), 40% irregular mixed
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

            vrv_opts.update({
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
            })
            stress_meta = {
                "spacing_stress": True,
                "target_bars_pattern": forced_pattern,
                "breaks_mode": breaks_mode,
                "spacingLinear": spacing_lin,
                "spacingNonLinear": spacing_nonlin,
                "evenNoteSpacing": even_note_spacing,
                "spacingSystem": spacing_system,
                "spacingStaff": spacing_staff,
                "justifyVertically": justify_vert,
                "staffLineWidth": staff_lw,
                "stemWidth": stem_w,
                "barLineWidth": bar_lw,
                "beamMaxSlope": beam_max_slope,
            }

        tk.setOptions(vrv_opts)
        if forced_pattern is not None:
            xml_str = inject_system_breaks_xml(xml_path, forced_pattern)
            loaded = tk.loadData(xml_str)
        else:
            loaded = tk.loadFile(str(xml_path))
        if not loaded:
            return {"score_id": score_id, "pages": [], "rejections": {"verovio_load_fail": 1}, "elapsed_s": round(time.time() - t0, 2)}

        n_pages = tk.getPageCount()
        if n_pages < 1:
            return {"score_id": score_id, "pages": [], "rejections": {"zero_pages": 1}, "elapsed_s": round(time.time() - t0, 2)}

        # Pass 1: Render SVGs for ALL pages to verify exact total measure count parity
        svgs: List[str] = []
        page_sys_counts: List[List[int]] = []
        total_svg_measures = 0
        for p in range(1, n_pages + 1):
            svg = tk.renderToSVG(p)
            sc = parse_verovio_svg_measure_counts(svg)
            total_svg_measures += sum(sc)
            if p <= max_pages:
                svgs.append(svg)
                page_sys_counts.append(sc)

        if total_svg_measures != total_xml_measures:
            return {
                "score_id": score_id,
                "pages": [],
                "rejections": {"measure_count_mismatch": 1},
                "elapsed_s": round(time.time() - t0, 2),
            }

        # Pass 2: For each page (up to max_pages), emit & validate per-system bekerns + PNG
        cur_m = 1
        pages_dir = out_dir / "pages"
        beks_dir = out_dir / "bekerns"
        pages_dir.mkdir(parents=True, exist_ok=True)
        beks_dir.mkdir(parents=True, exist_ok=True)

        for pi, (svg, sc) in enumerate(zip(svgs, page_sys_counts), start=1):
            if not sc:
                rej["empty_page"] += 1
                continue
            spans: List[Tuple[int, int]] = []
            for nm in sc:
                spans.append((cur_m, cur_m + nm - 1))
                cur_m += nm

            page_stem = f"{score_id}__vrv_{font}_{preset['name']}{'_sp' if spacing_stress else ''}_p{pi}"
            sys_bek_raws: List[str] = []
            page_tok_total = 0
            page_ok = True

            with tempfile.TemporaryDirectory() as td:
                td_p = Path(td)
                for si, (sm, em) in enumerate(spans):
                    s_stem = f"{page_stem}_s{si}"
                    bek_path, n_tok, err = multi_engrave.emit_and_validate_bekern(
                        xml_path, sm, em, td_p, s_stem,
                        aser_bin=aser_bin, min_tokens=1, max_tokens=1200,
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
                    if not check_prefix_state(toks):
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

            # Rasterize SVG -> crisp RGB PNG
            img = svg_to_rgb_png(svg, width_px=int(preset["width_px"]))
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


def main() -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path(r"D:\tmp\uni-verovio-iter10"))
    ap.add_argument("--pdmx-dir", type=Path, default=Path(r"D:\tmp\corpus10x\pdmx\quota-renders"))
    ap.add_argument("--stratum-tags", type=Path, default=Path(r"D:\tmp\stratum_tags_v2.json"))
    ap.add_argument("--big500-dir", type=Path, default=REPO_ROOT / "data-gen" / "score-corpus" / "big500-renders")
    ap.add_argument("--target-pages", type=int, default=3000)
    ap.add_argument("--max-pages-per-score", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261128)
    ap.add_argument("--spacing-stress", action="store_true", help="Randomize bar counts per line & spacing/engraving options")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    workers = min(4, max(1, int(args.workers)))
    aser_bin = multi_engrave.default_assembly_serialize_bin()
    if not aser_bin.is_file():
        raise FileNotFoundError(f"assembly-serialize not found: {aser_bin}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.out_dir / "_resume_state.jsonl"
    manifest_path = args.out_dir / "manifest.jsonl"

    policy, blocked = build_strict_eval_filter(REPO_ROOT)
    cands, sel_stats = select_candidates(
        args.pdmx_dir, args.stratum_tags, args.big500_dir, policy, blocked, seed=args.seed
    )
    print(f"[verovio-iter10] candidates={len(cands)} stats={sel_stats}", flush=True)

    done_scores: Set[str] = set()
    all_pages: List[Dict[str, Any]] = []
    if args.resume and state_path.is_file():
        for ln in state_path.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            try:
                rec = json.loads(ln)
                done_scores.add(rec["score_id"])
                for pg in rec.get("pages", []):
                    if Path(pg["image"]).is_file() and all(Path(b).is_file() for b in pg["bekerns"]):
                        all_pages.append(pg)
            except Exception:
                continue
        print(f"[verovio-iter10] resumed: {len(done_scores)} scores, {len(all_pages)} pages", flush=True)
    elif state_path.exists():
        state_path.unlink()

    jobs: List[Dict[str, Any]] = []
    for idx, c in enumerate(cands):
        if c["score_id"] in done_scores:
            continue
        font = VEROVIO_FONTS[idx % len(VEROVIO_FONTS)]
        preset = LAYOUT_PRESETS[(idx // len(VEROVIO_FONTS)) % len(LAYOUT_PRESETS)]
        jseed = int(hashlib.sha256(f"{args.seed}:{c['score_id']}:{font}:{int(args.spacing_stress)}".encode()).hexdigest()[:8], 16)
        jobs.append({
            "score_id": c["score_id"],
            "xml_path": c["xml_path"],
            "out_dir": str(args.out_dir),
            "font": font,
            "preset": preset,
            "seed": jseed,
            "max_pages": args.max_pages_per_score,
            "aser_bin": str(aser_bin),
            "spacing_stress": bool(args.spacing_stress),
        })

    t0 = time.time()
    rej_tot = collections.Counter()
    ji = 0
    in_flight: Set[concurrent.futures.Future] = set()

    if len(all_pages) < args.target_pages:
        with state_path.open("a", encoding="utf-8") as sf:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
                while len(in_flight) < workers and ji < len(jobs):
                    in_flight.add(pool.submit(process_one_score, jobs[ji]))
                    ji += 1

                completed = 0
                while in_flight:
                    done_set, in_flight = concurrent.futures.wait(
                        in_flight, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    for fut in done_set:
                        res = fut.result()
                        sf.write(json.dumps(res, separators=(",", ":")) + "\n")
                        sf.flush()
                        all_pages.extend(res.get("pages", []))
                        rej_tot.update(res.get("rejections", {}))
                        completed += 1
                        if completed % 40 == 0 or len(all_pages) >= args.target_pages:
                            by_font = collections.Counter(p["font"] for p in all_pages)
                            print(
                                f"[verovio-iter10 {time.time()-t0:6.1f}s] scores={completed} "
                                f"pages={len(all_pages)}/{args.target_pages} "
                                f"fonts={dict(by_font)} rej={dict(rej_tot.most_common(5))}",
                                flush=True,
                            )

                    while len(in_flight) < workers and ji < len(jobs) and len(all_pages) < args.target_pages:
                        in_flight.add(pool.submit(process_one_score, jobs[ji]))
                        ji += 1

    # Trim to target_pages if slightly over? Keep all valid pages!
    tmp_m = manifest_path.with_suffix(".jsonl.tmp")
    with tmp_m.open("w", encoding="utf-8") as f:
        for pg in all_pages:
            f.write(json.dumps(pg, ensure_ascii=False) + "\n")
    os.replace(tmp_m, manifest_path)

    by_font = collections.Counter(p["font"] for p in all_pages)
    by_layout = collections.Counter(p["layout"] for p in all_pages)
    print(f"[verovio-iter10] DONE -> {manifest_path}", flush=True)
    print(f"  total_pages={len(all_pages)} elapsed={time.time()-t0:.1f}s", flush=True)
    print(f"  by_font={dict(by_font)}", flush=True)
    print(f"  by_layout={dict(by_layout)}", flush=True)
    print(f"  rejections={dict(rej_tot)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
