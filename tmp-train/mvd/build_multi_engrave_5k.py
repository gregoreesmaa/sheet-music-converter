"""Build a ~5,000-strip stratified multi-engraver (Verovio + LilyPond) training corpus.

Strictly CPU-only (zero CUDA usage). Wraps ``multi_engrave.py`` across parallel
CPU workers with:
  - Stratified score selection across 5 musical domains (``piano``,
    ``string_quartet_chamber``, ``choral``, ``dense_classical``, ``folk``) from
    permissive ``D:\\tmp\\corpus10x\\pdmx\\quota-renders`` and
    ``data-gen/score-corpus/big500-renders``.
  - Strict eval-leakage & NC exclusion via ``ExclusionPolicy.from_repo()``
    (zero ``gate20`` / ``exclusion.json``, ``suite31``, ``suite31_hw``, or NC
    material; ``leaked_eval == 0`` verified both pre-selection and post-build).
  - Isolated per-job staging directories with hard subprocess timeouts and
    atomic ``os.replace`` promotion of validated strip PNGs + post-I1
    ``*-bekern.txt`` files into ``<out_dir>``.
  - Strict per-strip quality gates:
      1. ``bekern-decode`` round-trip passed (enforced by ``multi_engrave.py``)
         and LilyPond system-count-equality gate passed.
      2. ``0`` OOV tokens (every token in ``data.VOCAB``).
      3. ``0`` non-monotonic I1 voice bars (``recanon_i1.is_stream_canonical_i1``).
      4. Valid ``hybrid_staff.jsonl`` row with ``1 <= S <= 4`` and ``skip == False``.
  - Crash-safe append-only ``_resume_state.jsonl`` + ``--resume`` support.

Usage:
  .\\.venv\\Scripts\\python.exe tmp-train/mvd/build_multi_engrave_5k.py
  .\\.venv\\Scripts\\python.exe tmp-train/mvd/build_multi_engrave_5k.py --resume
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

MVD_DIR = Path(__file__).resolve().parent
REPO_ROOT = MVD_DIR.parent.parent
if str(MVD_DIR) not in sys.path:
    sys.path.insert(0, str(MVD_DIR))

import build_unlabeled_10k  # noqa: E402
import data  # noqa: E402
import multi_engrave  # noqa: E402
import recanon_i1  # noqa: E402

DEFAULT_OUT_DIR = Path(r"D:\tmp\multi-engrave-5k")
DEFAULT_PDMX_DIR = Path(r"D:\tmp\corpus10x\pdmx\quota-renders")
DEFAULT_STRATUM_TAGS = Path(r"D:\tmp\stratum_tags_v2.json")
DEFAULT_BIG500_DIR = REPO_ROOT / "data-gen" / "score-corpus" / "big500-renders"

VEROVIO_FONT_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("Leipzig", "Bravura"),
    ("Bravura", "Petaluma"),
    ("Petaluma", "Leipzig"),
)

STRATA_ORDER: Tuple[str, ...] = (
    "piano",
    "string_quartet_chamber",
    "choral",
    "dense_classical",
    "folk",
)

VOCAB_SET = frozenset(data.VOCAB)


@dataclass(frozen=True)
class CandidateScore:
    score_id: str
    xml_path: str
    stratum: str
    source_pool: str
    est_staves: int
    est_pages: int


def _classify_xml_head(xml_path: Path) -> str:
    """Fast heuristic instrument family classification from XML header."""
    try:
        with xml_path.open("r", encoding="utf-8", errors="ignore") as f:
            head = f.read(6000).lower()
    except OSError:
        return "other"
    if any(w in head for w in ("soprano", "alto", "tenor", "choir", "choral", "satb", "voice", "vocal")):
        return "choral"
    if any(w in head for w in ("violin", "viola", "cello", "violoncello", "quartet", "string")):
        return "string_chamber"
    if any(w in head for w in ("piano", "harpsichord", "organ", "clavichord", "keyboard")):
        return "piano"
    return "other"


def select_stratified_candidates(
    pdmx_dir: Path,
    stratum_tags_path: Path,
    big500_dir: Path,
    policy: build_unlabeled_10k.ExclusionPolicy,
    *,
    per_stratum: int = 120,
    seed: int = 20261123,
) -> Tuple[List[CandidateScore], Dict[str, Any]]:
    """Select an interleaved stratified list of clean permissive MusicXML scores."""
    rng = random.Random(seed)
    pools: Dict[str, List[CandidateScore]] = {s: [] for s in STRATA_ORDER}
    excluded_eval = 0
    excluded_nc = 0
    excluded_staves = 0

    tags: Dict[str, Any] = {}
    if stratum_tags_path.is_file():
        tags = json.loads(stratum_tags_path.read_text(encoding="utf-8"))

    # 1. Scan PDMX quota-renders using stratum_tags_v2.json
    if pdmx_dir.is_dir() and tags:
        for stem in sorted(tags.keys()):
            meta = tags[stem]
            xml_path = pdmx_dir / f"{stem}.musicxml"
            if not xml_path.is_file():
                continue
            if policy.is_forbidden_nc(xml_path):
                excluded_nc += 1
                continue
            if policy.is_leaked_eval(xml_path):
                excluded_eval += 1
                continue
            sig = meta.get("signals") or {}
            ns = int(sig.get("n_staves", 0) or 0)
            pg = int(sig.get("pages", 1) or 1)
            if not (1 <= ns <= 4):
                excluded_staves += 1
                continue
            raw_s = str(meta.get("stratum", ""))
            # Prefer 1..6 page scores (good strip yield without huge slow scores)
            if pg > 8:
                continue

            if raw_s == "piano-polyphony":
                pools["piano"].append(
                    CandidateScore(stem, str(xml_path), "piano", "pdmx", ns, pg)
                )
            elif raw_s == "folk":
                pools["folk"].append(
                    CandidateScore(stem, str(xml_path), "folk", "pdmx", ns, pg)
                )
            elif raw_s == "meter-key-change":
                pools["dense_classical"].append(
                    CandidateScore(stem, str(xml_path), "dense_classical", "pdmx", ns, pg)
                )
            elif raw_s == "chamber+other":
                # Sub-classify into choral vs string_quartet_chamber
                cat = _classify_xml_head(xml_path)
                if cat == "choral":
                    pools["choral"].append(
                        CandidateScore(stem, str(xml_path), "choral", "pdmx", ns, pg)
                    )
                else:
                    pools["string_quartet_chamber"].append(
                        CandidateScore(
                            stem, str(xml_path), "string_quartet_chamber", "pdmx", ns, pg
                        )
                    )

    # 2. Supplement dense_classical / string_quartet_chamber / choral from clean big500-renders
    if big500_dir.is_dir():
        for xml_path in sorted(big500_dir.glob("*.musicxml")):
            if policy.is_forbidden_nc(xml_path):
                excluded_nc += 1
                continue
            if policy.is_leaked_eval(xml_path):
                excluded_eval += 1
                continue
            cat = _classify_xml_head(xml_path)
            if cat == "choral":
                target_s = "choral"
            elif cat == "string_chamber":
                target_s = "string_quartet_chamber"
            elif cat == "piano":
                target_s = "piano"
            else:
                target_s = "dense_classical"
            pools[target_s].append(
                CandidateScore(xml_path.stem, str(xml_path), target_s, "big500", 2, 2)
            )

    # Shuffle each stratum deterministically, preferring 2..4 page scores slightly
    # earlier so each score yields a healthy number of strips.
    selected_by_stratum: Dict[str, List[CandidateScore]] = {}
    for s in STRATA_ORDER:
        items = pools[s]
        rng.shuffle(items)
        # Partition into sweet-spot (2..5 pages) and 1-page / 6..8-page so we get
        # rich multi-system diversity without starving 1-page folk tunes.
        multi = [c for c in items if 2 <= c.est_pages <= 5]
        other = [c for c in items if not (2 <= c.est_pages <= 5)]
        # Interleave 70% multi-page + 30% other
        ordered: List[CandidateScore] = []
        mi = oi = 0
        while len(ordered) < per_stratum and (mi < len(multi) or oi < len(other)):
            for _ in range(7):
                if mi < len(multi) and len(ordered) < per_stratum:
                    ordered.append(multi[mi])
                    mi += 1
            for _ in range(3):
                if oi < len(other) and len(ordered) < per_stratum:
                    ordered.append(other[oi])
                    oi += 1
        selected_by_stratum[s] = ordered

    # Round-robin interleave across the 5 strata so any prefix is balanced
    interleaved: List[CandidateScore] = []
    max_len = max((len(v) for v in selected_by_stratum.values()), default=0)
    for i in range(max_len):
        for s in STRATA_ORDER:
            if i < len(selected_by_stratum[s]):
                interleaved.append(selected_by_stratum[s][i])

    stats = {
        "pool_sizes": {s: len(pools[s]) for s in STRATA_ORDER},
        "candidate_sizes": {s: len(selected_by_stratum[s]) for s in STRATA_ORDER},
        "total_candidates": len(interleaved),
        "excluded_eval": excluded_eval,
        "excluded_nc": excluded_nc,
        "excluded_staves_gt4": excluded_staves,
    }
    return interleaved, stats


def _validate_and_promote_meta(
    meta_path: Path,
    stage_dir: Path,
    out_dir: Path,
    *,
    score_id: str,
    stratum: str,
    source_pool: str,
) -> Tuple[List[Dict[str, Any]], Counter]:
    """Validate every strip in ``meta_path`` and atomically move valid ones to ``out_dir``."""
    rej = Counter()
    valid_records: List[Dict[str, Any]] = []
    if not meta_path.is_file():
        rej["missing_meta"] += 1
        return valid_records, rej

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    engraver = str(meta.get("engraver", "unknown"))
    font = str(meta.get("font", "unknown"))

    if engraver == "lilypond" and meta.get("system_count_mismatch"):
        rej["lp_sys_count_mismatch"] += 1
        return valid_records, rej

    for page in meta.get("pages", []):
        for s in page.get("systems", []):
            strip_png_name = s.get("strip_png")
            bek_name = s.get("bekern_txt")
            srow = s.get("staff_row")

            if not strip_png_name or not bek_name or not srow:
                err = s.get("bekern_error") or "missing_bekern"
                rej[f"no_bekern:{str(err).split(':', 1)[0]}"] += 1
                continue

            if srow.get("skip"):
                rej["staff_row_skip"] += 1
                continue

            n_staves = int(srow.get("S", 0) or 0)
            if not (1 <= n_staves <= 4):
                rej[f"bad_S:{n_staves}"] += 1
                continue

            table = srow.get("table") or {}
            staves_tbl = table.get("staves") or []
            sp = float(srow.get("sp", 0.0) or 0.0)
            if len(staves_tbl) != n_staves or sp <= 0.0:
                rej["bad_staff_table"] += 1
                continue

            src_png = stage_dir / strip_png_name
            src_bek = stage_dir / bek_name
            if not src_png.is_file() or not src_bek.is_file():
                rej["missing_files"] += 1
                continue

            # 1. Read and validate bekern tokens (VOCAB + I1 monotonicity)
            raw_bek = src_bek.read_text(encoding="utf-8")
            tokens = raw_bek.split()
            if not tokens:
                rej["empty_bekern"] += 1
                continue

            oov = [t for t in tokens if t not in VOCAB_SET]
            if oov:
                rej["oov_tokens"] += 1
                continue

            if not recanon_i1.is_stream_canonical_i1(raw_bek):
                rej["non_monotonic_i1"] += 1
                continue

            # 2. Verify PNG header is readable and non-degenerate
            try:
                with Image.open(src_png) as im:
                    w, h = im.size
                if w < 32 or h < 24:
                    rej["tiny_strip"] += 1
                    continue
            except Exception:
                rej["corrupt_png"] += 1
                continue

            # 3. Promote atomically into out_dir
            dst_png = out_dir / strip_png_name
            dst_bek = out_dir / bek_name
            os.replace(src_png, dst_png)
            os.replace(src_bek, dst_bek)

            valid_records.append({
                "img": strip_png_name,
                "bekern": bek_name,
                "engraver": engraver,
                "font": font,
                "score_id": score_id,
                "stratum": stratum,
                "source_pool": source_pool,
                "S": n_staves,
                "n_tokens": len(tokens),
                "width": w,
                "height": h,
                "staff_row": srow,
            })

    return valid_records, rej


def run_single_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """Worker entrypoint: render one (score, engraver) job in an isolated staging dir."""
    score_id = job["score_id"]
    xml_path = job["xml_path"]
    stratum = job["stratum"]
    source_pool = job["source_pool"]
    engraver = job["engraver"]  # "verovio" or "lilypond"
    vfonts: List[str] = job.get("verovio_fonts", [])
    out_dir = Path(job["out_dir"])
    max_pages = int(job["max_pages"])
    timeout_s = int(job["timeout_s"])
    aser_bin = job["aser_bin"]

    job_key = f"{score_id}__{engraver}"
    stage_dir = out_dir / "_staging" / f"{job_key}_{os.getpid()}"
    if stage_dir.exists():
        shutil.rmtree(stage_dir, ignore_errors=True)
    stage_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""  # Strictly CPU-only!

    cmd = [
        sys.executable,
        str(MVD_DIR / "multi_engrave.py"),
        "--input", xml_path,
        "--out", str(stage_dir),
        "--engravers", engraver,
        "--max-pages", str(max_pages),
        "--no-staff-jsonl",
        "--aser-bin", aser_bin,
    ]
    if engraver == "verovio" and vfonts:
        cmd.extend(["--verovio-fonts", ",".join(vfonts)])

    error: Optional[str] = None
    valid_records: List[Dict[str, Any]] = []
    rej = Counter()

    try:
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
            check=False,
        )
        if res.returncode != 0:
            err_tail = (res.stderr or res.stdout or "").strip()[-240:]
            error = f"rc={res.returncode}:{err_tail}"
        else:
            xml_stem = Path(xml_path).stem
            if engraver == "verovio":
                meta_paths = [stage_dir / f"{xml_stem}__vrv_{f}_meta.json" for f in vfonts]
            else:
                meta_paths = [stage_dir / f"{xml_stem}__lp_meta.json"]
            for mp in meta_paths:
                recs, r_cnt = _validate_and_promote_meta(
                    mp,
                    stage_dir,
                    out_dir,
                    score_id=score_id,
                    stratum=stratum,
                    source_pool=source_pool,
                )
                valid_records.extend(recs)
                rej.update(r_cnt)
    except subprocess.TimeoutExpired:
        error = f"timeout:{timeout_s}s"
    except Exception as exc:
        error = f"exc:{type(exc).__name__}:{str(exc)[:160]}"
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)

    return {
        "job_key": job_key,
        "score_id": score_id,
        "xml_path": xml_path,
        "stratum": stratum,
        "source_pool": source_pool,
        "engraver": engraver,
        "verovio_fonts": vfonts,
        "elapsed_s": round(time.time() - t0, 2),
        "error": error,
        "valid_records": valid_records,
        "rejections": dict(rej),
    }


def _atomic_write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    lines = [json.dumps(r, separators=(",", ":")) for r in rows]
    tmp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _load_resume_state(state_path: Path) -> Dict[str, Dict[str, Any]]:
    done: Dict[str, Dict[str, Any]] = {}
    if not state_path.is_file():
        return done
    for ln in state_path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        try:
            rec = json.loads(ln)
            jk = rec.get("job_key")
            if jk:
                done[jk] = rec
        except json.JSONDecodeError:
            continue
    return done


def verify_output_corpus(
    out_dir: Path,
    policy: build_unlabeled_10k.ExclusionPolicy,
    all_job_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Full end-to-end verification of every emitted strip, bekern, and staff row."""
    staff_path = out_dir / "hybrid_staff.jsonl"
    staff_rows: Dict[str, Dict[str, Any]] = {}
    if staff_path.is_file():
        for ln in staff_path.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                r = json.loads(ln)
                staff_rows[r["img"]] = r

    pngs = sorted(p for p in out_dir.glob("*.png"))
    beks = sorted(p for p in out_dir.glob("*-bekern.txt"))

    png_names = {p.name for p in pngs}
    bek_stems = {p.name[: -len("-bekern.txt")] for p in beks}

    leaked_eval = 0
    nc_count = 0
    oov_total = 0
    oov_files = 0
    total_bars = 0
    bad_i1_bars = 0
    bad_i1_files = 0
    bad_s_rows = 0
    missing_pair = 0

    by_engraver = Counter()
    by_font = Counter()
    by_stratum = Counter()
    by_staves = Counter()
    by_source_pool = Counter()
    unique_scores_with_strips = set()

    # Map strip img -> metadata from job results
    img_meta: Dict[str, Dict[str, Any]] = {}
    for res in all_job_results.values():
        xml_p = res.get("xml_path", "")
        if xml_p:
            if policy.is_leaked_eval(xml_p):
                leaked_eval += 1
            if policy.is_forbidden_nc(xml_p):
                nc_count += 1
        for rec in res.get("valid_records", []):
            img_meta[rec["img"]] = rec

    for p in pngs:
        if policy.is_leaked_eval(p):
            leaked_eval += 1
        if policy.is_forbidden_nc(p):
            nc_count += 1
        stem = p.stem
        if stem not in bek_stems:
            missing_pair += 1
        srow = staff_rows.get(p.name)
        if not srow or srow.get("skip") or not (1 <= int(srow.get("S", 0)) <= 4):
            bad_s_rows += 1
        else:
            by_staves[int(srow["S"])] += 1

        meta = img_meta.get(p.name)
        if meta:
            by_engraver[meta["engraver"]] += 1
            by_font[f"{meta['engraver']}/{meta['font']}"] += 1
            by_stratum[meta["stratum"]] += 1
            by_source_pool[meta["source_pool"]] += 1
            unique_scores_with_strips.add(meta["score_id"])

    for bp in beks:
        stem = bp.name[: -len("-bekern.txt")]
        if f"{stem}.png" not in png_names:
            missing_pair += 1
        raw = bp.read_text(encoding="utf-8")
        toks = raw.split()
        file_oov = sum(1 for t in toks if t not in VOCAB_SET)
        if file_oov:
            oov_total += file_oov
            oov_files += 1
        _, body = recanon_i1.split_leading_headers(toks)
        file_bad = False
        for m_toks, _ in recanon_i1.split_measures(body):
            total_bars += 1
            if not recanon_i1.is_measure_monotonic(m_toks):
                bad_i1_bars += 1
                file_bad = True
        if file_bad:
            bad_i1_files += 1

    bad_bar_pct = (100.0 * bad_i1_bars / total_bars) if total_bars else 0.0
    return {
        "total_strip_pngs": len(pngs),
        "total_bekern_files": len(beks),
        "total_staff_jsonl_rows": len(staff_rows),
        "unique_scores_with_strips": len(unique_scores_with_strips),
        "by_engraver": dict(by_engraver),
        "by_font": dict(sorted(by_font.items())),
        "by_stratum": dict(sorted(by_stratum.items())),
        "by_staves": {str(k): v for k, v in sorted(by_staves.items())},
        "by_source_pool": dict(sorted(by_source_pool.items())),
        "leaked_eval": leaked_eval,
        "nc_count": nc_count,
        "oov_tokens": oov_total,
        "oov_files": oov_files,
        "total_bars": total_bars,
        "bad_i1_bars": bad_i1_bars,
        "bad_i1_files": bad_i1_files,
        "bad_i1_bar_pct": round(bad_bar_pct, 4),
        "bad_s_rows": bad_s_rows,
        "missing_pair": missing_pair,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build ~5k stratified Verovio + LilyPond multi-engraver strip corpus (CPU-only)."
    )
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--pdmx-dir", type=Path, default=DEFAULT_PDMX_DIR)
    ap.add_argument("--stratum-tags", type=Path, default=DEFAULT_STRATUM_TAGS)
    ap.add_argument("--big500-dir", type=Path, default=DEFAULT_BIG500_DIR)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--target-verovio", type=int, default=3300)
    ap.add_argument("--target-lilypond", type=int, default=1700)
    ap.add_argument("--per-stratum-pool", type=int, default=160)
    ap.add_argument("--max-pages", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20261123)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--aser-bin", type=Path, default=None)
    args = ap.parse_args()

    t_start = time.time()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "_resume_state.jsonl"

    aser_bin = args.aser_bin or multi_engrave.default_assembly_serialize_bin()
    if not aser_bin.is_file():
        raise FileNotFoundError(f"assembly-serialize binary not found: {aser_bin}")
    lp_bin, m2l_bin = multi_engrave.find_lilypond_bins()
    if lp_bin is None or m2l_bin is None:
        raise FileNotFoundError(f"LilyPond not found: lp={lp_bin}, m2l={m2l_bin}")

    policy = build_unlabeled_10k.ExclusionPolicy.from_repo(REPO_ROOT)

    candidates, sel_stats = select_stratified_candidates(
        args.pdmx_dir,
        args.stratum_tags,
        args.big500_dir,
        policy,
        per_stratum=args.per_stratum_pool,
        seed=args.seed,
    )
    print(
        f"[select] candidates={len(candidates)} "
        f"(excluded_eval={sel_stats['excluded_eval']}, excluded_nc={sel_stats['excluded_nc']}, "
        f"excluded_staves_gt4={sel_stats['excluded_staves_gt4']})",
        flush=True,
    )
    print(f"[select] by_stratum={sel_stats['candidate_sizes']}", flush=True)

    done_jobs: Dict[str, Dict[str, Any]] = {}
    if args.resume or args.verify_only:
        done_jobs = _load_resume_state(state_path)
        print(f"[resume] loaded {len(done_jobs)} completed jobs from {state_path}", flush=True)
    else:
        # Clean fresh state file if starting fresh
        if state_path.exists():
            state_path.unlink()

    # Count already-completed valid strips on disk
    vrv_count = 0
    lp_count = 0
    for rec in done_jobs.values():
        kept = []
        for r in rec.get("valid_records", []):
            if (out_dir / r["img"]).is_file() and (out_dir / r["bekern"]).is_file():
                kept.append(r)
        rec["valid_records"] = kept
        if rec.get("engraver") == "verovio":
            vrv_count += len(kept)
        elif rec.get("engraver") == "lilypond":
            lp_count += len(kept)

    print(
        f"[init] existing valid strips: verovio={vrv_count}/{args.target_verovio}, "
        f"lilypond={lp_count}/{args.target_lilypond}",
        flush=True,
    )

    if not args.verify_only and (vrv_count < args.target_verovio or lp_count < args.target_lilypond):
        # Build per-engraver queues in interleaved stratified order
        vrv_queue: List[Dict[str, Any]] = []
        lp_queue: List[Dict[str, Any]] = []
        for idx, cand in enumerate(candidates):
            v_key = f"{cand.score_id}__verovio"
            if v_key not in done_jobs:
                vrv_queue.append({
                    "score_id": cand.score_id,
                    "xml_path": cand.xml_path,
                    "stratum": cand.stratum,
                    "source_pool": cand.source_pool,
                    "engraver": "verovio",
                    "verovio_fonts": list(VEROVIO_FONT_PAIRS[idx % len(VEROVIO_FONT_PAIRS)]),
                    "out_dir": str(out_dir),
                    "max_pages": args.max_pages,
                    "timeout_s": args.timeout,
                    "aser_bin": str(aser_bin),
                })
            l_key = f"{cand.score_id}__lilypond"
            if l_key not in done_jobs:
                lp_queue.append({
                    "score_id": cand.score_id,
                    "xml_path": cand.xml_path,
                    "stratum": cand.stratum,
                    "source_pool": cand.source_pool,
                    "engraver": "lilypond",
                    "verovio_fonts": [],
                    "out_dir": str(out_dir),
                    "max_pages": args.max_pages,
                    "timeout_s": args.timeout,
                    "aser_bin": str(aser_bin),
                })

        vi = 0
        li = 0
        in_flight: Dict[concurrent.futures.Future, str] = {}
        in_flight_vrv = 0
        in_flight_lp = 0
        completed_since_log = 0
        total_errors = Counter()

        def _next_job() -> Optional[Dict[str, Any]]:
            nonlocal vi, li
            need_vrv = (vrv_count < args.target_verovio) and (vi < len(vrv_queue))
            need_lp = (lp_count < args.target_lilypond) and (li < len(lp_queue))
            if not need_vrv and not need_lp:
                return None
            # Balance dispatch proportionally to remaining deficit so both finish together
            if need_vrv and need_lp:
                v_frac = vrv_count / max(1, args.target_verovio)
                l_frac = lp_count / max(1, args.target_lilypond)
                if l_frac <= v_frac:
                    j = lp_queue[li]
                    li += 1
                    return j
                j = vrv_queue[vi]
                vi += 1
                return j
            if need_lp:
                j = lp_queue[li]
                li += 1
                return j
            j = vrv_queue[vi]
            vi += 1
            return j

        with state_path.open("a", encoding="utf-8") as sf:
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
                # Prime worker pool
                while len(in_flight) < args.workers:
                    nxt = _next_job()
                    if nxt is None:
                        break
                    fut = pool.submit(run_single_job, nxt)
                    in_flight[fut] = nxt["engraver"]
                    if nxt["engraver"] == "verovio":
                        in_flight_vrv += 1
                    else:
                        in_flight_lp += 1

                while in_flight:
                    done_set, _ = concurrent.futures.wait(
                        in_flight.keys(),
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for fut in done_set:
                        eng = in_flight.pop(fut)
                        if eng == "verovio":
                            in_flight_vrv -= 1
                        else:
                            in_flight_lp -= 1
                        res = fut.result()
                        done_jobs[res["job_key"]] = res
                        sf.write(json.dumps(res, separators=(",", ":")) + "\n")
                        sf.flush()

                        n_add = len(res.get("valid_records", []))
                        if eng == "verovio":
                            vrv_count += n_add
                        else:
                            lp_count += n_add
                        if res.get("error"):
                            total_errors[f"{eng}:{res['error'].split(':', 1)[0]}"] += 1

                        completed_since_log += 1
                        if completed_since_log >= 25 or (
                            vrv_count >= args.target_verovio and lp_count >= args.target_lilypond
                        ):
                            completed_since_log = 0
                            elapsed = time.time() - t_start
                            print(
                                f"[progress {elapsed:6.1f}s] jobs={len(done_jobs)} "
                                f"verovio={vrv_count}/{args.target_verovio} "
                                f"lilypond={lp_count}/{args.target_lilypond} "
                                f"total={vrv_count + lp_count} errs={dict(total_errors)}",
                                flush=True,
                            )

                    # Top up pool while targets remain unmet
                    while len(in_flight) < args.workers:
                        nxt = _next_job()
                        if nxt is None:
                            break
                        fut = pool.submit(run_single_job, nxt)
                        in_flight[fut] = nxt["engraver"]
                        if nxt["engraver"] == "verovio":
                            in_flight_vrv += 1
                        else:
                            in_flight_lp += 1

        # Clean up empty staging directory
        staging_root = out_dir / "_staging"
        if staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)

    # Rebuild hybrid_staff.jsonl atomically from all verified strips on disk
    staff_by_img: Dict[str, Dict[str, Any]] = {}
    rej_totals = Counter()
    job_errors = Counter()
    for res in done_jobs.values():
        for k, v in (res.get("rejections") or {}).items():
            rej_totals[f"{res.get('engraver')}:{k}"] += int(v)
        if res.get("error"):
            job_errors[f"{res.get('engraver')}:{res['error'].split(':', 1)[0]}"] += 1
        for r in res.get("valid_records", []):
            img_name = r["img"]
            bek_name = r["bekern"]
            if (out_dir / img_name).is_file() and (out_dir / bek_name).is_file():
                staff_by_img[img_name] = r["staff_row"]

    sorted_staff_rows = [staff_by_img[k] for k in sorted(staff_by_img.keys())]
    _atomic_write_jsonl(out_dir / "hybrid_staff.jsonl", sorted_staff_rows)

    # Full end-to-end verification pass on disk
    verify = verify_output_corpus(out_dir, policy, done_jobs)
    elapsed_total = round(time.time() - t_start, 2)

    summary = {
        "out_dir": str(out_dir),
        "seed": args.seed,
        "workers": args.workers,
        "max_pages": args.max_pages,
        "target_verovio": args.target_verovio,
        "target_lilypond": args.target_lilypond,
        "elapsed_s": elapsed_total,
        "selection": sel_stats,
        "jobs_completed": len(done_jobs),
        "job_errors": dict(sorted(job_errors.items())),
        "strip_rejections": dict(sorted(rej_totals.items())),
        "verification": verify,
    }
    _atomic_write_json(out_dir / "summary.json", summary)

    print("\n=== MULTI-ENGRAVER 5K SUMMARY ===", flush=True)
    print(json.dumps(summary["verification"], indent=2), flush=True)

    # Hard assertions on all invariants
    assert verify["leaked_eval"] == 0, f"EVAL LEAKAGE DETECTED: {verify['leaked_eval']}"
    assert verify["nc_count"] == 0, f"NC MATERIAL DETECTED: {verify['nc_count']}"
    assert verify["oov_tokens"] == 0, f"OOV TOKENS DETECTED: {verify['oov_tokens']}"
    assert verify["bad_i1_bars"] == 0, f"NON-MONOTONIC I1 BARS DETECTED: {verify['bad_i1_bars']}"
    assert verify["bad_s_rows"] == 0, f"INVALID S ROWS DETECTED: {verify['bad_s_rows']}"
    assert verify["missing_pair"] == 0, f"MISSING PNG/BEKERN PAIRS: {verify['missing_pair']}"
    print(f"[OK] All invariants passed in {elapsed_total}s -> {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
