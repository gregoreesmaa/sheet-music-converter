#!/usr/bin/env python3
"""Build the Iter 14 `bekern+v2` Phase A Full-Page Manifest (100% CPU-only).

Never imports torch or touches CUDA. Never overwrites `D:\\tmp\\uni_fullpage_iter13_clean.jsonl`.

Pipeline:
  1. Reads `D:\\tmp\\uni_fullpage_iter13_clean.jsonl` (47,477 rows).
  2. For every row whose source MusicXML is traceable on disk (`corpus10x`, `t32farm`,
     `corpus100k`, `uni-verovio*`, `uni-osmd*`, `uni-printed-c-8vb-iter8`,
     `uni-printed-rare-iter8`, and `photo-degrade-iter13-2k` rows whose `source_image`
     maps to one of those), when `--reexport-v2` is passed, re-exports per-system
     `bekern+v2` targets using `assembly-serialize.exe bekern --v2` (parallelized +
     cached on disk in `--v2-cache-dir`), falling back gracefully to existing v1
     `bekerns` for non-XML rows or any export edge case (`v1 ⊂ v2`, 0 OOV).
  3. Appends the 250 synthetic `neg_control` rows (`D:\\tmp\\neg-controls-iter14\\manifest.jsonl`,
     empty target `""` -> `<eos>`), plus optionally any `rism_pseudo` JSONL
     (`--rism-pseudo-jsonl`).
  4. Verifies:
       - `0` eval overlap against ALL 45 `suite31_hw` slugs
         (`data-gen/score-corpus/suite31/suite31_hw_manifest.json` + `mozart-k458`)
       - `0` eval overlap against ALL 125 `omr-northstar` pages
         (`D:\\tmp\\northstar-eval-pack\\manifest.json`)
       - `0` OOV tokens against `bekern+v2` Phase A vocab (`VOCAB_SIZE = 265`,
         `len(data.VOCAB) == 261` + PAD/BOS/EOS/`<sys>`)
       - Exact token & v2-token census in stats JSON.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import collections
import concurrent.futures
import hashlib
import json
import random
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

MVD_DIR = Path(__file__).resolve().parent
REPO_ROOT = MVD_DIR.parent.parent
if str(MVD_DIR) not in sys.path:
    sys.path.insert(0, str(MVD_DIR))

import align_hw_work_to_pages as ahw  # noqa: E402
import data as mvd_data  # noqa: E402
import render_verovio_dataset as rvd  # noqa: E402

VOCAB_SET = frozenset(mvd_data.VOCAB)
V2A_NEW_SET = frozenset(mvd_data._BEKERN_V2A_ADD)
UNI_VOCAB_SIZE = len(mvd_data.VOCAB) + 4  # PAD, BOS, EOS, <sys> == 265

DEFAULT_BASE = Path(r"D:\tmp\uni_fullpage_iter13_clean.jsonl")
DEFAULT_NEG_CONTROLS = Path(r"D:\tmp\neg-controls-iter14\manifest.jsonl")
DEFAULT_OUT = Path(r"D:\tmp\uni_fullpage_iter14_v2a.jsonl")
DEFAULT_STATS = Path(r"D:\tmp\uni_fullpage_iter14_v2a_stats.json")
DEFAULT_V2_CACHE = Path(r"D:\tmp\bekern-v2a-cache")
DEFAULT_ASSEMBLY_BIN = REPO_ROOT / "converter" / "assembly-serialize" / "target" / "release" / "assembly-serialize.exe"
SUITE31_HW_MANIFEST = REPO_ROOT / "data-gen" / "score-corpus" / "suite31" / "suite31_hw_manifest.json"
NORTHSTAR_MANIFEST = Path(r"D:\tmp\northstar-eval-pack\manifest.json")

_PAGE_NUM_RE = re.compile(r"^(.+)-(\d+)$")
_PDMX_QM_RE = re.compile(r"^pdmx_(Qm[A-Za-z0-9]+)__")
_STITCHED_HW_MARKERS: Tuple[str, ...] = ("1s2s", "stitched", "concat", "synth_page", "composed")


def _norm_path_key(p: str) -> str:
    return os.path.normcase(os.path.normpath(p))


def classify_bucket_iter14(img_path: str, domain: str = "", bucket: str = "") -> str:
    b_in = str(bucket or "").strip().lower()
    dom = str(domain or "").strip().lower()
    p_str = str(img_path).replace("\\", "/").lower()
    if b_in == "neg_control" or dom == "neg_control" or "neg-controls" in p_str:
        return "neg_control"
    if b_in == "rism_pseudo" or dom == "rism_pseudo" or "rism_pseudo" in p_str:
        return "rism_pseudo"
    if dom == "real_hw" or "real_hw" in p_str or "debussy" in p_str:
        if any(tok in p_str or tok in dom for tok in _STITCHED_HW_MARKERS):
            return "real_hw_stitched"
        return "real_hw_archival"
    if dom in ("hw", "synth_hw") or "hw-sim" in p_str or "hw_sim" in p_str:
        return "synth_hw"
    if "c-8vb" in p_str or "c_8vb" in p_str or "rare" in p_str or dom == "rare_printed":
        return "rare_printed"
    if (
        "verovio" in p_str
        or "osmd" in p_str
        or "spacing" in p_str
        or "corpus100k" in p_str
        or dom == "multi_renderer_printed"
    ):
        return "multi_renderer_printed"
    return "legacy_printed"


def load_suite31_hw_slugs(manifest_path: Path = SUITE31_HW_MANIFEST) -> List[str]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing suite31_hw_manifest.json: {manifest_path}")
    d = json.loads(manifest_path.read_text(encoding="utf-8"))
    slugs = [str(p["slug"]).strip() for p in d.get("pieces", []) if str(p.get("slug", "")).strip()]
    if len(slugs) != 45:
        raise ValueError(f"Expected 45 suite31_hw slugs, got {len(slugs)}")
    return slugs


def load_northstar_eval_index(manifest_path: Path = NORTHSTAR_MANIFEST) -> Dict[str, Any]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing northstar manifest.json: {manifest_path}")
    d = json.loads(manifest_path.read_text(encoding="utf-8"))
    pages = d.get("pages", [])
    if len(pages) != 125:
        raise ValueError(f"Expected 125 northstar pages, got {len(pages)}")
    img_keys: Set[str] = set()
    slugs: Set[str] = set()
    for p in pages:
        ip = str(p.get("image_path", "")).strip()
        if ip:
            img_keys.add(_norm_path_key(ip))
        for field in ("page_id", "work_id", "paired_anchor_id"):
            val = str(p.get(field, "")).strip()
            if not val:
                continue
            ns = ahw._norm_slug(val)
            for prefix in ("cdr-", "als-", "rwp-", "hwm-", "work-", "anchor-"):
                if ns.startswith(prefix):
                    ns = ns[len(prefix):]
            if len(ns) >= 5:
                slugs.add(ns)
                if ns.startswith("hw-"):
                    slugs.add(ns[3:])
    return {
        "n_pages": len(pages),
        "img_keys": img_keys,
        "slugs": sorted(slugs),
    }


def _match_eval_overlap(
    path_str: str,
    suite31_slugs: Sequence[str],
    northstar_idx: Dict[str, Any],
) -> Optional[str]:
    if _norm_path_key(path_str) in northstar_idx["img_keys"]:
        return f"northstar_img:{path_str}"
    norm_text = ahw._norm_slug(path_str)
    padded = f"-{norm_text}-"
    if "mozart-k458" in norm_text or "k-458" in norm_text or "kv-458" in norm_text or "kv458" in norm_text:
        return "suite31:mozart-k458"
    for slug in suite31_slugs:
        ns = ahw._norm_slug(slug)
        bare = ns[3:] if ns.startswith("hw-") else ns
        for cand in (ns, bare):
            if len(cand) >= 5 and f"-{cand}-" in padded:
                return f"suite31:{slug}"
    for ns_slug in northstar_idx["slugs"]:
        if len(ns_slug) >= 5 and f"-{ns_slug}-" in padded:
            return f"northstar:{ns_slug}"
    return None


class XmlResolver:
    """Fast lookup tables mapping iter13 full-page images -> (musicxml_path, spans_or_page)."""

    def __init__(self) -> None:
        self.c100k_score_to_xml: Dict[str, str] = {}
        self.img_to_spans: Dict[str, Tuple[Optional[str], List[Tuple[int, int]]]] = {}
        self.pdmx_xml_dirs: List[Path] = [
            Path(r"D:\tmp\corpus10x\pdmx\renders"),
            Path(r"D:\tmp\corpus10x\pdmx\quota-renders"),
            Path(r"D:\tmp\corpus10x\pdmx\quota-renders-leland"),
        ]
        self._loaded = False

    def load_indices(self) -> None:
        if self._loaded:
            return
        t0 = time.time()
        c100k_m = Path(r"D:\tmp\corpus100k\corpus100k_manifest.jsonl")
        if c100k_m.is_file():
            with c100k_m.open("r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    d = json.loads(s)
                    sid = d.get("score_id")
                    pth = d.get("path", "")
                    if sid and pth and str(pth).lower().endswith((".musicxml", ".xml", ".mxl")):
                        self.c100k_score_to_xml[str(sid)] = str(pth)

        span_manifests = [
            Path(r"D:\tmp\corpus100k\rendered_iter10\manifest.jsonl"),
            Path(r"D:\tmp\uni-verovio-iter10\manifest.jsonl"),
            Path(r"D:\tmp\uni-verovio-spacing-iter10\manifest.jsonl"),
            Path(r"D:\tmp\uni-osmd-iter10\manifest.jsonl"),
            Path(r"D:\tmp\uni-osmd-spacing-iter10\manifest.jsonl"),
        ]
        for mp in span_manifests:
            if not mp.is_file():
                continue
            with mp.open("r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    d = json.loads(s)
                    img = d.get("image")
                    spans = d.get("measure_spans")
                    if not img or not spans:
                        continue
                    norm_spans = [(int(a), int(b)) for a, b in spans]
                    sid = d.get("score_id")
                    self.img_to_spans[_norm_path_key(str(img))] = (str(sid) if sid else None, norm_spans)
        self._loaded = True
        print(
            f"[build-iter14] Loaded XML lookup indices: "
            f"c100k_xml={len(self.c100k_score_to_xml):,}, "
            f"span_pages={len(self.img_to_spans):,} ({time.time() - t0:.1f}s)",
            flush=True,
        )

    def _find_pdmx_xml(self, qm_hash: str) -> Optional[Path]:
        for d in self.pdmx_xml_dirs:
            cand = d / f"{qm_hash}.musicxml"
            if cand.is_file():
                return cand
        return None

    def resolve(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Return {'xml': Path, 'mode': 'spans'|'page', 'spans': [...], 'page': int} or None."""
        img_str = str(row.get("source_image") or row.get("image") or "")
        if not img_str:
            return None
        img_p = Path(img_str)
        k = _norm_path_key(img_str)

        # 1. Manifest-indexed measure_spans (corpus100k, verovio*, osmd*)
        if k in self.img_to_spans:
            sid, spans = self.img_to_spans[k]
            xml_p: Optional[Path] = None
            if sid and sid in self.c100k_score_to_xml:
                cand = Path(self.c100k_score_to_xml[sid])
                if cand.is_file():
                    xml_p = cand
            else:
                m = _PDMX_QM_RE.match(img_p.name)
                if m:
                    xml_p = self._find_pdmx_xml(m.group(1))
            if xml_p is not None and spans:
                return {"xml": xml_p, "mode": "spans", "spans": spans, "page": None}
            return None

        # 2. Sibling `<base>.musicxml` for `<base>-<page>.png`
        # Covers: t32farm, corpus10x (quota-renders / quota-renders-leland), c-8vb, rare
        m = _PAGE_NUM_RE.match(img_p.stem)
        if m:
            base_stem, page_str = m.group(1), m.group(2)
            sib_xml = img_p.parent / f"{base_stem}.musicxml"
            if sib_xml.is_file():
                return {"xml": sib_xml, "mode": "page", "spans": None, "page": int(page_str)}

        return None


def _run_system_breaks(assembly_bin: str, xml_path: str) -> Tuple[str, Optional[Dict[int, List[Tuple[int, int]]]]]:
    """Run `assembly-serialize.exe system-breaks --input <xml>` and group spans by 1-based page."""
    try:
        proc = subprocess.run(
            [assembly_bin, "system-breaks", "--input", xml_path],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return xml_path, None
        arr = json.loads(proc.stdout)
        by_page: Dict[int, List[Tuple[int, int]]] = collections.defaultdict(list)
        for item in arr:
            pg = int(item["page"])
            # Support start_measure/end_measure (assembly-serialize system-breaks) + fallbacks
            if "start_measure" in item and "end_measure" in item:
                s, e = int(item["start_measure"]), int(item["end_measure"])
            elif "start" in item and "end" in item:
                s, e = int(item["start"]), int(item["end"])
            elif "first_measure" in item and "last_measure" in item:
                s, e = int(item["first_measure"]), int(item["last_measure"])
            elif "measure_start" in item and "measure_end" in item:
                s, e = int(item["measure_start"]), int(item["measure_end"])
            else:
                return xml_path, None
            by_page[pg].append((s, e))
        return xml_path, dict(by_page)
    except Exception:
        return xml_path, None


def _extract_mxl_if_needed(xml_path: str, tmp_dir: Path, stem_base: str) -> Optional[Tuple[str, Optional[Path]]]:
    """If `xml_path` is a compressed `.mxl` archive, extract its root MusicXML to a temp file."""
    if not xml_path.lower().endswith(".mxl"):
        return xml_path, None
    try:
        with zipfile.ZipFile(xml_path, "r") as zf:
            target_member: Optional[str] = None
            if "META-INF/container.xml" in zf.namelist():
                cdata = zf.read("META-INF/container.xml").decode("utf-8", errors="ignore")
                m = re.search(r'full-path="([^"]+)"', cdata)
                if m and m.group(1) in zf.namelist():
                    target_member = m.group(1)
            if target_member is None:
                for name in zf.namelist():
                    nl = name.lower()
                    if not nl.startswith("meta-inf/") and nl.endswith((".musicxml", ".xml")):
                        target_member = name
                        break
            if target_member is None:
                return None
            out_xml = tmp_dir / f"{stem_base}_tmp.musicxml"
            out_xml.write_bytes(zf.read(target_member))
            return str(out_xml), out_xml
    except Exception:
        return None


def _export_page_v2(
    job: Tuple[int, str, str, List[Tuple[int, int]], str, str],
) -> Tuple[int, Optional[List[str]], str]:
    """Re-export all systems of one page using `assembly-serialize.exe bekern --v2`."""
    row_idx, assembly_bin, xml_path, spans, cache_dir, page_key = job
    # Shard cache into 256 subdirs to avoid huge single-dir slowdown on Windows NTFS
    h = hashlib.sha1(page_key.encode("utf-8")).hexdigest()
    sub_dir = Path(cache_dir) / h[:2]
    sub_dir.mkdir(parents=True, exist_ok=True)
    stem_base = f"{h[:16]}"

    # Fast check: if all cached outputs already exist, skip any .mxl extraction
    out_paths = [str(sub_dir / f"{stem_base}_s{s_idx}-bekern.txt") for s_idx in range(len(spans))]
    if all(Path(p).is_file() and Path(p).stat().st_size > 0 for p in out_paths):
        return row_idx, out_paths, "ok"

    extracted = _extract_mxl_if_needed(xml_path, sub_dir, stem_base)
    if extracted is None:
        return row_idx, None, "mxl_extract_failed"
    eff_xml, tmp_xml = extracted

    try:
        for s_idx, (m_start, m_end) in enumerate(spans):
            stem = f"{stem_base}_s{s_idx}"
            out_file = sub_dir / f"{stem}-bekern.txt"
            if not (out_file.is_file() and out_file.stat().st_size > 0):
                try:
                    proc = subprocess.run(
                        [
                            assembly_bin,
                            "bekern",
                            "--v2",
                            "--input",
                            eff_xml,
                            "--measures",
                            f"{m_start}..{m_end}",
                            "--out",
                            str(sub_dir),
                            "--stem",
                            stem,
                        ],
                        capture_output=True,
                        text=True,
                        timeout=30,
                        check=False,
                    )
                    if proc.returncode != 0 or not out_file.is_file():
                        return row_idx, None, "export_cmd_failed"
                except Exception:
                    return row_idx, None, "export_exception"
        return row_idx, out_paths, "ok"
    finally:
        if tmp_xml is not None:
            try:
                tmp_xml.unlink(missing_ok=True)
            except Exception:
                pass


def _inspect_bekern_file(bp: str) -> Tuple[str, bool, bool, int, List[str]]:
    """Return (path, exists, vocab_ok, tok_len, v2_new_toks_found)."""
    p = Path(bp)
    if not p.is_file():
        return bp, False, False, 0, []
    try:
        toks = mvd_data.load_bekern(p)
    except ValueError:
        return bp, True, False, 0, []
    v2_hits: List[str] = []
    for t in toks:
        if t not in VOCAB_SET:
            return bp, True, False, len(toks), []
        if t in V2A_NEW_SET:
            v2_hits.append(t)
    return bp, True, True, len(toks), v2_hits


def _check_file_exists(path_str: str) -> Tuple[str, bool]:
    return path_str, Path(path_str).is_file()


def _load_jsonl(path: Path, limit: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def build_iter14_manifest(
    base_path: Path,
    neg_controls_path: Optional[Path],
    rism_pseudo_paths: Sequence[Path],
    out_path: Path,
    stats_path: Path,
    *,
    reexport_v2: bool = False,
    assembly_bin: Path = DEFAULT_ASSEMBLY_BIN,
    v2_cache_dir: Path = DEFAULT_V2_CACHE,
    limit: int = 0,
    max_tokens: int = 3840,
    max_systems: int = 16,
    seed: int = 20261127,
    workers: int = 16,
) -> Dict[str, Any]:
    t0 = time.time()
    if _norm_path_key(str(out_path)) == _norm_path_key(str(base_path)):
        raise ValueError(f"Refusing to overwrite base manifest: {base_path}")
    if not base_path.is_file():
        raise FileNotFoundError(f"Base Iter-13 manifest not found: {base_path}")
    if reexport_v2 and not assembly_bin.is_file():
        raise FileNotFoundError(f"assembly-serialize.exe not found: {assembly_bin}")

    assert UNI_VOCAB_SIZE == 265, f"Expected bekern+v2 Phase A VOCAB_SIZE=265, got {UNI_VOCAB_SIZE}"

    suite31_slugs = load_suite31_hw_slugs()
    northstar_idx = load_northstar_eval_index()
    policy, blocked = rvd.build_strict_eval_filter(REPO_ROOT)

    print(f"[build-iter14] Loading base manifest: {base_path} (limit={limit or 'ALL'}) ...", flush=True)
    base_rows = _load_jsonl(base_path, limit=limit)
    print(f"[build-iter14] Loaded {len(base_rows):,} base rows ({time.time() - t0:.1f}s)", flush=True)

    # Optional v2 re-export on traceable XML rows
    reexport_stats: collections.Counter = collections.Counter()
    if reexport_v2:
        resolver = XmlResolver()
        resolver.load_indices()
        resolved: Dict[int, Dict[str, Any]] = {}
        xmls_needing_breaks: Set[str] = set()
        for idx, row in enumerate(base_rows):
            res = resolver.resolve(row)
            if res is None:
                reexport_stats["non_xml_v1_fallback"] += 1
                continue
            resolved[idx] = res
            if res["mode"] == "page":
                xmls_needing_breaks.add(str(res["xml"]))

        print(
            f"[build-iter14] Traceable XML rows: {len(resolved):,} / {len(base_rows):,} "
            f"(unique XMLs needing system-breaks: {len(xmls_needing_breaks):,})",
            flush=True,
        )

        # 1. Parallel system-breaks for page-mode XMLs
        breaks_by_xml: Dict[str, Dict[int, List[Tuple[int, int]]]] = {}
        if xmls_needing_breaks:
            tb = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(_run_system_breaks, str(assembly_bin), xp) for xp in sorted(xmls_needing_breaks)]
                for fut in concurrent.futures.as_completed(futs):
                    xp, by_pg = fut.result()
                    if by_pg is not None:
                        breaks_by_xml[xp] = by_pg
            print(
                f"[build-iter14] Resolved system-breaks for {len(breaks_by_xml):,} / "
                f"{len(xmls_needing_breaks):,} XMLs ({time.time() - tb:.1f}s)",
                flush=True,
            )

        # 2. Build export jobs
        jobs: List[Tuple[int, str, str, List[Tuple[int, int]], str, str]] = []
        v2_cache_dir.mkdir(parents=True, exist_ok=True)
        for idx, res in resolved.items():
            xml_str = str(res["xml"])
            if res["mode"] == "spans":
                spans = res["spans"]
            else:
                by_pg = breaks_by_xml.get(xml_str)
                if not by_pg or res["page"] not in by_pg:
                    reexport_stats["fallback_missing_system_breaks"] += 1
                    continue
                spans = by_pg[res["page"]]
            if not spans or len(spans) > max_systems:
                reexport_stats["fallback_spans_out_of_range"] += 1
                continue
            # Ensure system count matches existing bekern count to guarantee visual alignment
            orig_n = len(base_rows[idx].get("bekerns", []))
            if orig_n > 0 and len(spans) != orig_n:
                reexport_stats["fallback_system_count_mismatch"] += 1
                continue
            page_key = f"{xml_str}|spans={spans}"
            jobs.append((idx, str(assembly_bin), xml_str, spans, str(v2_cache_dir), page_key))

        print(f"[build-iter14] Exporting {len(jobs):,} pages via `assembly-serialize bekern --v2` (workers={workers}) ...", flush=True)
        te = time.time()
        exported_map: Dict[int, List[str]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            for done_i, (r_idx, out_bks, status) in enumerate(ex.map(_export_page_v2, jobs), start=1):
                if out_bks is not None:
                    exported_map[r_idx] = out_bks
                else:
                    reexport_stats[f"fallback_{status}"] += 1
                if done_i % 2000 == 0 or done_i == len(jobs):
                    print(f"  [export-v2] {done_i:,}/{len(jobs):,} ({time.time() - te:.1f}s)", flush=True)

        # Validate exported v2 files (0 OOV, <= max_tokens) before swapping in
        v2_beks = {b for bks in exported_map.values() for b in bks}
        v2_bek_info: Dict[str, Tuple[bool, bool, int, List[str]]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            for bp, exists, vok, tlen, v2h in ex.map(_inspect_bekern_file, v2_beks):
                v2_bek_info[bp] = (exists, vok, tlen, v2h)

        for r_idx, out_bks in exported_map.items():
            raw_len = 0
            ok = True
            for b in out_bks:
                exists, vok, tlen, _ = v2_bek_info[b]
                if not exists or not vok or tlen == 0:
                    ok = False
                    break
                raw_len += tlen
            full_len = raw_len + max(0, len(out_bks) - 1) + 2
            if not ok:
                reexport_stats["fallback_v2_oov_or_empty"] += 1
                continue
            if full_len > max_tokens:
                reexport_stats["fallback_v2_gt_max_tokens"] += 1
                continue
            base_rows[r_idx] = dict(base_rows[r_idx])
            base_rows[r_idx]["bekerns"] = out_bks
            base_rows[r_idx]["bekern_version"] = "v2a"
            reexport_stats["reexported_v2"] += 1
    else:
        reexport_stats["v1_kept_no_reexport_flag"] = len(base_rows)

    # Load negative controls
    neg_rows: List[Dict[str, Any]] = []
    if neg_controls_path is not None and neg_controls_path.is_file():
        neg_limit = min(10, limit) if limit > 0 else 0
        neg_rows = _load_jsonl(neg_controls_path, limit=neg_limit)
        print(f"[build-iter14] Loaded {len(neg_rows):,} neg_control rows from {neg_controls_path}", flush=True)

    # Load optional RISM pseudo-labels
    rism_rows: List[Dict[str, Any]] = []
    for rp in rism_pseudo_paths:
        if rp.is_file():
            loaded = _load_jsonl(rp, limit=min(10, limit) if limit > 0 else 0)
            for r in loaded:
                r.setdefault("domain", "rism_pseudo")
                r.setdefault("bucket", "rism_pseudo")
            rism_rows.extend(loaded)
            print(f"[build-iter14] Loaded {len(loaded):,} rism_pseudo rows from {rp}", flush=True)

    # Combine & deduplicate by normalized image path
    combined_raw = base_rows + neg_rows + rism_rows
    dedup_rows: List[Dict[str, Any]] = []
    seen_imgs: Set[str] = set()
    dup_dropped = 0
    for row in combined_raw:
        img = str(row.get("image", ""))
        key = _norm_path_key(img)
        if not img or key in seen_imgs:
            dup_dropped += 1
            continue
        seen_imgs.add(key)
        clean_row: Dict[str, Any] = {
            "image": img,
            "bekerns": [str(b) for b in row.get("bekerns", [])],
            "domain": str(row.get("domain", "printed")),
        }
        if "bucket" in row:
            clean_row["bucket"] = str(row["bucket"])
        if "bekern_version" in row:
            clean_row["bekern_version"] = str(row["bekern_version"])
        else:
            clean_row["bekern_version"] = "empty" if clean_row["domain"] == "neg_control" else "v1"
        if "neg_type" in row:
            clean_row["neg_type"] = str(row["neg_type"])
        for extra_k in ("source_image", "source_stratum", "photo_profile", "photo_seed", "photo_strength"):
            if extra_k in row:
                clean_row[extra_k] = row[extra_k]
        dedup_rows.append(clean_row)

    # Full parallel disk + vocab + token-length + v2-token census
    unique_imgs = {r["image"] for r in dedup_rows}
    unique_beks = {b for r in dedup_rows for b in r["bekerns"]}
    print(
        f"[build-iter14] Verifying {len(unique_imgs):,} unique images and "
        f"{len(unique_beks):,} unique bekerns (workers={workers}) ...",
        flush=True,
    )
    img_exists: Dict[str, bool] = {}
    bek_info: Dict[str, Tuple[bool, bool, int, List[str]]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for p, ok in ex.map(_check_file_exists, unique_imgs):
            img_exists[p] = ok
        for bp, exists, vok, tlen, v2h in ex.map(_inspect_bekern_file, unique_beks):
            bek_info[bp] = (exists, vok, tlen, v2h)

    # Filter & strict QA check
    out_rows: List[Dict[str, Any]] = []
    dropped: collections.Counter = collections.Counter()
    eval_hits: List[str] = []

    for row in dedup_rows:
        img = row["image"]
        bks = row["bekerns"]
        dom = row["domain"]
        is_neg = dom == "neg_control" or row.get("bucket") == "neg_control"

        if not img or not bks or not img_exists.get(img, False):
            dropped["missing_image_or_empty_bekerns"] += 1
            continue
        if len(bks) > max_systems:
            dropped["gt_max_systems"] += 1
            continue

        ov = _match_eval_overlap(img, suite31_slugs, northstar_idx)
        if ov is not None or (not is_neg and rvd.is_path_blocked(img, policy, blocked)):
            dropped["eval_or_nc_blocked"] += 1
            if ov is not None:
                eval_hits.append(ov)
            continue

        bek_blocked = False
        missing_bek = False
        oov_bek = False
        empty_non_neg = False
        raw_tok_len = 0
        for b in bks:
            bov = _match_eval_overlap(b, suite31_slugs, northstar_idx)
            if bov is not None or (not is_neg and rvd.is_path_blocked(b, policy, blocked)):
                bek_blocked = True
                if bov is not None:
                    eval_hits.append(bov)
                break
            exists, vok, tlen, _ = bek_info[b]
            if not exists:
                missing_bek = True
                break
            if not vok:
                oov_bek = True
                break
            if not is_neg and tlen == 0:
                empty_non_neg = True
                break
            raw_tok_len += tlen

        if bek_blocked:
            dropped["eval_or_nc_blocked"] += 1
            continue
        if missing_bek:
            dropped["missing_bekern"] += 1
            continue
        if oov_bek:
            dropped["oov_bekern"] += 1
            continue
        if empty_non_neg:
            dropped["empty_non_neg_bekern"] += 1
            continue

        full_tok_len = raw_tok_len + max(0, len(bks) - 1) + 2
        if full_tok_len > max_tokens:
            dropped["gt_max_tokens"] += 1
            continue

        out_rows.append(row)

    rng = random.Random(seed)
    rng.shuffle(out_rows)

    # Hard final QA pass + exact token stats
    domain_counts: collections.Counter = collections.Counter()
    bucket_counts: collections.Counter = collections.Counter()
    version_counts: collections.Counter = collections.Counter()
    v2_token_counts: collections.Counter = collections.Counter()
    total_tokens = 0
    total_systems = 0
    verify_imgs: Set[str] = set()
    final_eval_overlap = 0

    for row in out_rows:
        img = row["image"]
        bks = row["bekerns"]
        dom = row["domain"]
        bkt = classify_bucket_iter14(img, dom, row.get("bucket", ""))
        k = _norm_path_key(img)
        assert k not in verify_imgs, f"QA FAIL: duplicate image {img}"
        verify_imgs.add(k)
        if _match_eval_overlap(img, suite31_slugs, northstar_idx) is not None:
            final_eval_overlap += 1
        for b in bks:
            if _match_eval_overlap(b, suite31_slugs, northstar_idx) is not None:
                final_eval_overlap += 1
            _, vok, tlen, v2h = bek_info[b]
            assert vok, f"QA FAIL: OOV in {b}"
            total_tokens += tlen
            for t in v2h:
                v2_token_counts[t] += 1
        total_systems += len(bks)
        domain_counts[dom] += 1
        bucket_counts[bkt] += 1
        version_counts[row.get("bekern_version", "v1")] += 1

    assert final_eval_overlap == 0, f"QA FAIL: {final_eval_overlap} eval overlaps!"

    # Atomic write
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_out.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_out, out_path)

    total = len(out_rows)
    stats: Dict[str, Any] = {
        "inputs": {
            "base_manifest": str(base_path),
            "base_rows_loaded": len(base_rows),
            "neg_controls_manifest": str(neg_controls_path) if neg_controls_path else None,
            "neg_controls_loaded": len(neg_rows),
            "rism_pseudo_manifests": [str(p) for p in rism_pseudo_paths],
            "rism_pseudo_loaded": len(rism_rows),
            "reexport_v2": bool(reexport_v2),
            "limit": int(limit),
        },
        "output_manifest": str(out_path),
        "seed": seed,
        "vocab_size": UNI_VOCAB_SIZE,
        "base_vocab_len": len(mvd_data.VOCAB),
        "total_rows": total,
        "total_systems": total_systems,
        "total_tokens": total_tokens,
        "mean_tokens_per_page": round(total_tokens / max(1, total), 2),
        "duplicates_removed": dup_dropped,
        "reexport_v2_summary": dict(sorted(reexport_stats.items())),
        "filter_drops": dict(sorted(dropped.items())),
        "counts_by_domain": {
            k: {"rows": int(v), "pct": round(100.0 * v / max(1, total), 2)}
            for k, v in sorted(domain_counts.items())
        },
        "counts_by_bucket": {
            k: {"rows": int(v), "pct": round(100.0 * v / max(1, total), 2)}
            for k, v in sorted(bucket_counts.items())
        },
        "counts_by_bekern_version": dict(sorted(version_counts.items())),
        "v2a_new_token_counts": dict(sorted(v2_token_counts.items(), key=lambda x: -x[1])),
        "v2a_new_tokens_total": int(sum(v2_token_counts.values())),
        "qa_verification": {
            "vocab_size": UNI_VOCAB_SIZE,
            "suite31_hw_slugs_checked": len(suite31_slugs),
            "northstar_pages_checked": northstar_idx["n_pages"],
            "eval_overlap": final_eval_overlap,
            "duplicate_images": 0,
            "missing_files": 0,
            "oov_tokens": 0,
            "passed": True,
        },
        "elapsed_sec": round(time.time() - t0, 2),
    }

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_stats = stats_path.with_suffix(stats_path.suffix + ".tmp")
    tmp_stats.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp_stats, stats_path)

    print("\n" + "=" * 78, flush=True)
    print(f"[build-iter14] DONE -> {out_path} ({stats['elapsed_sec']:.1f}s)", flush=True)
    print(f"  Total rows   : {total:,} (systems={total_systems:,}, tokens={total_tokens:,}, mean={stats['mean_tokens_per_page']}/pg)", flush=True)
    print(f"  By domain    : { {k: v['rows'] for k, v in stats['counts_by_domain'].items()} }", flush=True)
    print(f"  By bucket    : { {k: v['rows'] for k, v in stats['counts_by_bucket'].items()} }", flush=True)
    print(f"  By version   : {stats['counts_by_bekern_version']}", flush=True)
    print(f"  Re-export v2 : {stats['reexport_v2_summary']}", flush=True)
    print(f"  v2a new toks : {stats['v2a_new_tokens_total']:,} across {len(stats['v2a_new_token_counts'])} types: {stats['v2a_new_token_counts']}", flush=True)
    print(
        f"  QA           : 0 / {len(suite31_slugs)} suite31_hw | 0 / {northstar_idx['n_pages']} northstar | "
        f"0 dup | 0 missing | 0 OOV (VOCAB_SIZE={UNI_VOCAB_SIZE}) | PASS",
        flush=True,
    )
    print("=" * 78, flush=True)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Build Iter 14 bekern+v2 Phase A full-page manifest (CPU-only).")
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--neg-controls", type=Path, default=DEFAULT_NEG_CONTROLS)
    ap.add_argument("--rism-pseudo-jsonl", type=Path, action="append", default=[])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--stats", type=Path, default=None)
    ap.add_argument("--reexport-v2", action="store_true")
    ap.add_argument("--assembly-bin", type=Path, default=DEFAULT_ASSEMBLY_BIN)
    ap.add_argument("--v2-cache-dir", type=Path, default=DEFAULT_V2_CACHE)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=3840)
    ap.add_argument("--max-systems", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20261127)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    stats_path = args.stats if args.stats is not None else args.out.with_name(args.out.stem + "_stats.json")
    build_iter14_manifest(
        base_path=args.base,
        neg_controls_path=args.neg_controls,
        rism_pseudo_paths=args.rism_pseudo_jsonl,
        out_path=args.out,
        stats_path=stats_path,
        reexport_v2=args.reexport_v2,
        assembly_bin=args.assembly_bin,
        v2_cache_dir=args.v2_cache_dir,
        limit=args.limit,
        max_tokens=args.max_tokens,
        max_systems=args.max_systems,
        seed=args.seed,
        workers=args.workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
