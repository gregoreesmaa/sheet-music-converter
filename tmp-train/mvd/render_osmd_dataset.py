#!/usr/bin/env python3
"""Render a variation-rich, leak-free OSMD / VexFlow full-page dataset for UniOMR Iter 10.

Strictly CPU-only (CUDA_VISIBLE_DEVICES=""), runs headless Chromium via
`D:\\tmp\\osmd-runner\\render_osmd_batch.mjs` (Playwright + local OSMD bundle,
zero network dependency).

Visual variation:
  - VexFlow engraving fonts: Bravura, Gonville, Petaluma
  - OSMD drawingParameters: default, compact, compacttight
  - Viewport widths: 900, 1050, 1200, 1350 px (1.5x DPR -> 1350..2025 px crisp PNGs)
  - Zoom: 0.78..1.12
  - Page margins: 20..56 px
  - Occasional titles (20%)

Exact <sys> alignment guarantee:
  - Extracts exact measure spans directly from OSMD's `osmd.graphic.MeasureList`
    (`ParentMusicSystem.Id` + Y-coordinate).
  - Enforces `osmd_nMeasures == total_musicxml_measures`.
  - Emits per-system bekern+ via `assembly-serialize bekern --measures S..E`,
    validated by `bekern-decode`, 0 OOV against `data.VOCAB`, `PrefixState(strict=False)`,
    and `recanon_i1.is_stream_canonical_i1`.
  - Strict 0-overlap against suite31_hw_manifest.json, gate20/exclusion.json,
    mozart-k458, and non-commercial sources.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from PIL import Image

MVD_DIR = Path(__file__).resolve().parent
REPO_ROOT = MVD_DIR.parent.parent
if str(MVD_DIR) not in sys.path:
    sys.path.insert(0, str(MVD_DIR))

import data as mvd_data  # noqa: E402
import multi_engrave  # noqa: E402
import recanon_i1  # noqa: E402
import render_verovio_dataset as rvd  # noqa: E402

OSMD_FONTS: Tuple[str, ...] = ("Bravura", "Gonville", "Petaluma")
OSMD_DRAWING_PARAMS: Tuple[str, ...] = ("default", "compact", "compacttight")
OSMD_VIEWPORTS: Tuple[int, ...] = (900, 1050, 1200, 1350)

VOCAB_SET = frozenset(mvd_data.VOCAB)
MAX_PAGE_TOKENS = 3800


def select_osmd_candidates(
    pdmx_dir: Path,
    stratum_tags_path: Path,
    big500_dir: Path,
    max_measures: int,
    seed: int,
) -> List[Dict[str, Any]]:
    policy, blocked = rvd.build_strict_eval_filter(REPO_ROOT)
    rng = random.Random(seed)
    cands: List[Dict[str, Any]] = []

    tags: Dict[str, Any] = {}
    if stratum_tags_path.is_file():
        tags = json.loads(stratum_tags_path.read_text(encoding="utf-8"))

    # Prefer 1..2 page scores (4..max_measures bars, 1..4 staves) so the entire score
    # renders as one crisp, readable full page in OSMD with zero pagination drift.
    if pdmx_dir.is_dir():
        for xml_path in sorted(pdmx_dir.glob("*.musicxml")):
            if rvd.is_path_blocked(str(xml_path), policy, blocked):
                continue
            stem = xml_path.stem
            meta = tags.get(stem, {})
            sig = meta.get("signals") or {}
            ns = int(sig.get("n_staves", 0) or 0)
            pg = int(sig.get("pages", 1) or 1)
            if ns > 4 or pg > 2:
                continue
            cands.append({
                "score_id": f"pdmx_{stem}",
                "xml_path": str(xml_path),
            })

    if big500_dir.is_dir():
        for xml_path in sorted(big500_dir.glob("*.musicxml")):
            if rvd.is_path_blocked(str(xml_path), policy, blocked):
                continue
            cands.append({
                "score_id": f"big500_{xml_path.stem}",
                "xml_path": str(xml_path),
            })

    rng.shuffle(cands)

    # Filter by exact MusicXML measure count (4..max_measures)
    filtered: List[Dict[str, Any]] = []
    for c in cands:
        nm = rvd.count_xml_measures(Path(c["xml_path"]))
        if 4 <= nm <= max_measures:
            c["xml_measures"] = nm
            filtered.append(c)
        if len(filtered) >= 2600:
            break
    return filtered


def main() -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path(r"D:\tmp\uni-osmd-iter10"))
    ap.add_argument("--pdmx-dir", type=Path, default=Path(r"D:\tmp\corpus10x\pdmx\quota-renders"))
    ap.add_argument("--stratum-tags", type=Path, default=Path(r"D:\tmp\stratum_tags_v2.json"))
    ap.add_argument("--big500-dir", type=Path, default=REPO_ROOT / "data-gen" / "score-corpus" / "big500-renders")
    ap.add_argument("--target-pages", type=int, default=1200)
    ap.add_argument("--max-measures", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261129)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    aser_bin = multi_engrave.default_assembly_serialize_bin()
    if not aser_bin.is_file():
        raise FileNotFoundError(f"assembly-serialize not found: {aser_bin}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pages_dir = args.out_dir / "pages"
    beks_dir = args.out_dir / "bekerns"
    pages_dir.mkdir(parents=True, exist_ok=True)
    beks_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.out_dir / "manifest.jsonl"
    state_path = args.out_dir / "_osmd_state.jsonl"

    t0 = time.time()
    cands = select_osmd_candidates(
        args.pdmx_dir, args.stratum_tags, args.big500_dir, args.max_measures, args.seed
    )
    print(f"[osmd-iter10] selected {len(cands)} single-page candidates (4..{args.max_measures} bars) in {time.time()-t0:.1f}s", flush=True)

    done_ids: Set[str] = set()
    valid_pages: List[Dict[str, Any]] = []
    if args.resume and state_path.is_file():
        for ln in state_path.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            try:
                rec = json.loads(ln)
                done_ids.add(rec["score_id"])
                if rec.get("valid") and Path(rec["row"]["image"]).is_file() and all(Path(b).is_file() for b in rec["row"]["bekerns"]):
                    valid_pages.append(rec["row"])
            except Exception:
                continue
        print(f"[osmd-iter10] resumed: {len(done_ids)} attempted, {len(valid_pages)} valid pages", flush=True)
    elif state_path.exists():
        state_path.unlink()

    pending = [c for c in cands if c["score_id"] not in done_ids]
    cand_by_id = {c["score_id"]: c for c in cands}
    rej = collections.Counter()

    idx = 0
    with state_path.open("a", encoding="utf-8") as sf:
        while len(valid_pages) < args.target_pages and idx < len(pending):
            batch = pending[idx : idx + args.batch_size]
            idx += len(batch)

            node_jobs = []
            for j_i, c in enumerate(batch):
                g_idx = len(done_ids) + idx - len(batch) + j_i
                font = OSMD_FONTS[g_idx % len(OSMD_FONTS)]
                dp = OSMD_DRAWING_PARAMS[(g_idx // len(OSMD_FONTS)) % len(OSMD_DRAWING_PARAMS)]
                vw = OSMD_VIEWPORTS[(g_idx // (len(OSMD_FONTS) * len(OSMD_DRAWING_PARAMS))) % len(OSMD_VIEWPORTS)]
                jseed = int(hashlib.sha256(f"{args.seed}:{c['score_id']}".encode()).hexdigest()[:8], 16)
                rng = random.Random(jseed)
                zoom = round(rng.uniform(0.78, 1.10), 2)
                margin_px = rng.randint(20, 52)
                draw_title = rng.random() < 0.20
                stem = f"{c['score_id']}__osmd_{font}_{dp}"
                node_jobs.append({
                    "score_id": c["score_id"],
                    "xml_path": c["xml_path"],
                    "png_path": str(pages_dir / f"{stem}.png"),
                    "drawing_params": dp,
                    "font": font,
                    "viewport_w": vw,
                    "zoom": zoom,
                    "margin_px": margin_px,
                    "draw_title": draw_title,
                    "max_height_px": 2400,
                })

            with tempfile.TemporaryDirectory() as td:
                td_p = Path(td)
                jobs_json = td_p / "jobs.json"
                res_jsonl = td_p / "results.jsonl"
                jobs_json.write_text(json.dumps(node_jobs), encoding="utf-8")

                proc = subprocess.run(
                    ["node", r"D:\tmp\osmd-runner\render_osmd_batch.mjs", str(jobs_json), str(res_jsonl)],
                    capture_output=True,
                    text=True,
                    timeout=900,
                    check=False,
                )
                if proc.stdout:
                    for line in proc.stdout.strip().splitlines():
                        print(f"  {line}", flush=True)
                if proc.returncode != 0:
                    print(f"[osmd-iter10] WARN node rc={proc.returncode}: {(proc.stderr or '')[-240:]}", flush=True)

                if not res_jsonl.is_file():
                    continue

                for ln in res_jsonl.read_text(encoding="utf-8").splitlines():
                    if not ln.strip():
                        continue
                    r = json.loads(ln)
                    sid = r["score_id"]
                    c = cand_by_id[sid]
                    if not r.get("ok"):
                        rej[f"node_{(r.get('error') or 'err').split(':', 1)[0]}"] += 1
                        sf.write(json.dumps({"score_id": sid, "valid": False, "error": r.get("error")}) + "\n")
                        continue

                    png_path = Path(r["png_path"])
                    if int(r["nMeasures"]) != int(c["xml_measures"]):
                        rej["measure_count_mismatch"] += 1
                        png_path.unlink(missing_ok=True)
                        sf.write(json.dumps({"score_id": sid, "valid": False, "error": "measure_mismatch"}) + "\n")
                        continue

                    spans = [(int(a), int(b)) for a, b in r["spans"]]
                    xml_path = Path(c["xml_path"])
                    stem = png_path.stem
                    sys_raws: List[str] = []
                    tok_total = 0
                    page_ok = True
                    err_reason = ""

                    with tempfile.TemporaryDirectory() as btd:
                        btd_p = Path(btd)
                        for si, (sm, em) in enumerate(spans):
                            bek_path, n_tok, err = multi_engrave.emit_and_validate_bekern(
                                xml_path, sm, em, btd_p, f"{stem}_s{si}",
                                aser_bin=aser_bin, min_tokens=1, max_tokens=1200,
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
                        sf.write(json.dumps({"score_id": sid, "valid": False, "error": err_reason or "too_many_tokens"}) + "\n")
                        continue

                    # Verify PNG readable
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
                    }
                    valid_pages.append(row)
                    sf.write(json.dumps({"score_id": sid, "valid": True, "row": row}, separators=(",", ":")) + "\n")
                sf.flush()

            by_font = collections.Counter(p["font"] for p in valid_pages)
            print(
                f"[osmd-iter10 {time.time()-t0:6.1f}s] valid={len(valid_pages)}/{args.target_pages} "
                f"fonts={dict(by_font)} rej={dict(rej.most_common(5))}",
                flush=True,
            )

    tmp_m = manifest_path.with_suffix(".jsonl.tmp")
    with tmp_m.open("w", encoding="utf-8") as f:
        for pg in valid_pages:
            f.write(json.dumps(pg, ensure_ascii=False) + "\n")
    os.replace(tmp_m, manifest_path)

    by_font = collections.Counter(p["font"] for p in valid_pages)
    by_dp = collections.Counter(p["drawing_params"] for p in valid_pages)
    print(f"[osmd-iter10] DONE -> {manifest_path}", flush=True)
    print(f"  total_pages={len(valid_pages)} elapsed={time.time()-t0:.1f}s", flush=True)
    print(f"  by_font={dict(by_font)}", flush=True)
    print(f"  by_drawing_params={dict(by_dp)}", flush=True)
    print(f"  rejections={dict(rej)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
