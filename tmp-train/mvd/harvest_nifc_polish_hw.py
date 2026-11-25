#!/usr/bin/env python3
"""NIFC Polish Heritage Handwritten Manuscript Corpus Harvester (CPU-only).

Harvests full-work HANDWRITTEN MANUSCRIPTS from `pl-wnifc/humdrum-polish-scores`
(`CC-BY 4.0` diplomatic `**kern` transcriptions) paired with verified scans from:

  1. Staatsbibliothek zu Berlin (`d-b`, 34 works across 2 codices):
     - Explicitly verified per-PPN `PDM 1.0` (`https://creativecommons.org/publicdomain/mark/1.0/`)
       via SBB METS XML (`content.staatsbibliothek-berlin.de/dc/{ppn}.mets.xml`).
     - IIIF v2 canvas downloads (`<= 1800 px` JPEG Q=92) with polite per-host rate limiting.
  2. NIFC Polish Music Sources (`polish.musicsources.pl/.../rekopisy/<id>`, 4,969 manuscripts):
     - Explicitly user-approved Option 2: 16th-19th c. Polish heritage manuscripts digitized
       under EU POPC 2.3.2 / DMP Open Access (EU Directive 2019/1024 Art. 14).
     - Strictly filtered to `/rekopisy/` (handwritten manuscripts ONLY; excludes `/druki-muzyczne/`,
       `/klocki-introligatorskie/`, `gallica.bnf.fr`, and any item with `NC`/`InC`/restricted
       text in `rights_note`).
     - Per-item `!!!YEM:` check requires explicit `CC-BY 4.0` (excludes placeholder/empty headers).
     - Downloads `MEDIUM_SIZE` RGB JPEGs via `api.nifc.pl/v2` + `repozytorium.nifc.pl`, resized to
       `<= 1800 px` JPEG Q=92 with fast CPU blank/cover filtering.

Other archives audited & excluded by design:
  - `bc.bdsandomierz.pl` (`pl-sa`, 1,162 works / 273 editions): explicit per-item
    `Domena publiczna (Public domain)` + `rękopis muzyczny`, but serves indirect multi-page
    DjVu (`image/x.djvu`) only (no JPEG/PNG/PDF endpoint and no local `ddjvu` binary).
  - `polona2.pl` / `polona.pl` (568 works): printed editions (`druk muzyczny`), excluded.
  - `gallica.bnf.fr` (37 works): hard-blocked (non-commercial).

Architecture:
  - Stage A (ProcessPoolExecutor): GIL-free parallel `**kern -> MusicXML -> post-I1 bekern ->
    bekern-decode rc=0` conversion across CPU cores.
  - Stage B (ThreadPoolExecutor): Parallel scan downloads with per-host throttling (polite SBB
    rate limiting, fast NIFC `MEDIUM_SIZE` streams), CPU blank/cover filter, atomic tmp+rename,
    and optional hardlinking into `D:\\tmp\\unlabeled-10k-real\\nifc_hw\\images\\`.

Usage:
    .\\.venv\\Scripts\\python.exe tmp-train/mvd/harvest_nifc_polish_hw.py ^
        --limit 150 --max-pages-per-work 8 --workers 6 --resume
"""

from __future__ import annotations

import os

# Hard-disable CUDA/GPU before any library imports (CPU + network only)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import argparse
import collections
import concurrent.futures
import io
import json
import logging
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import requests
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

USER_AGENT = "sheet-music-converter-research/1.0 (NIFC-Polish-HW-harvester; CPU-only)"
DEFAULT_REPO_DIR = Path(r"D:\tmp\nifc-polish-scores-repo")
DEFAULT_OUT_DIR = Path(r"D:\tmp\nifc-polish-hw")
DEFAULT_UNLABELED_DIR = Path(r"D:\tmp\unlabeled-10k-real\nifc_hw\images")
DEFAULT_ASSEMBLY_BIN = (
    REPO_ROOT
    / "converter"
    / "assembly-serialize"
    / "target"
    / "release"
    / "assembly-serialize.exe"
)
SUITE31_MANIFEST = REPO_ROOT / "data-gen" / "score-corpus" / "suite31" / "suite31_manifest.json"
SUITE31_HW_MANIFEST = REPO_ROOT / "data-gen" / "score-corpus" / "suite31" / "suite31_hw_manifest.json"

NIFC_API_MS_URL = "https://api.nifc.pl/v2/index.php/manuscripts/get_manuscript_by_id/{ms_id}"
SBB_IIIF_URL = "https://content.staatsbibliothek-berlin.de/dc/{ppn}/manifest"
SBB_METS_URL = "https://content.staatsbibliothek-berlin.de/dc/{ppn}.mets.xml"

REJECT_LICENSE_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"\bby-nc\b", re.I),
    re.compile(r"\bcc[- ]?by[- ]?nc\b", re.I),
    re.compile(r"non[- ]?commercial", re.I),
    re.compile(r"niekomercyj", re.I),
    re.compile(r"rightsstatements\.org/vocab/inc", re.I),
    re.compile(r"\bin copyright\b", re.I),
    re.compile(r"\ball rights reserved\b", re.I),
    re.compile(r"wszelkie prawa zastrze", re.I),
    re.compile(r"\brestricted\b", re.I),
    re.compile(r"gallica\.bnf\.fr", re.I),
)

ACCEPT_PDM_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"creativecommons\.org/publicdomain/mark/1\.0", re.I),
    re.compile(r"creativecommons\.org/publicdomain/zero/1\.0", re.I),
    re.compile(r"rightsstatements\.org/vocab/noc-oklr", re.I),
)


@dataclass(frozen=True)
class Candidate:
    work_id: str
    archive: str
    rel_path: str
    krn_path: str
    source_kind: str  # "sbb_pdm" | "nifc_popc2_pd"
    source_id: str    # PPN for sbb_pdm, manuscript numeric id for nifc_popc2_pd
    url_scan: str
    page_hint: int    # 1-based page hint (0 if none)
    composer: str
    title: str
    shelfmark: str
    yem_license: str


def normalize_str(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
    return s


def slugify(s: str, max_len: int = 90) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", s).strip("_")
    return s[:max_len] or "work"


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def hardlink_or_copy(src: Path, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return True
    try:
        os.link(src, dst)
        return True
    except OSError:
        try:
            shutil.copy2(src, dst)
            return True
        except OSError:
            return False


def load_eval_guard() -> Tuple[Set[str], Set[str]]:
    slugs: Set[str] = set()
    titles: Set[str] = set()
    for mf in (SUITE31_MANIFEST, SUITE31_HW_MANIFEST):
        if not mf.exists():
            continue
        data = json.loads(mf.read_text(encoding="utf-8"))
        for piece in data.get("pieces", []):
            sl = str(piece.get("slug") or "").strip()
            if sl:
                slugs.add(normalize_str(sl))
            for k in ("title", "work_title", "description", "source_url"):
                val = piece.get(k)
                if isinstance(val, str) and len(val.strip()) >= 4:
                    titles.add(normalize_str(val))
            for subk in ("hw_meta", "meta", "source"):
                sub = piece.get(subk)
                if isinstance(sub, dict):
                    for vk in ("title", "slug", "rism_id", "ppn", "manuscript_id"):
                        v = sub.get(vk)
                        if isinstance(v, str) and v.strip():
                            titles.add(normalize_str(v))
    return slugs, titles


def is_eval_overlap(cand: Candidate, eval_slugs: Set[str], eval_titles: Set[str]) -> bool:
    wid_norm = normalize_str(cand.work_id)
    if wid_norm in eval_slugs:
        return True
    if cand.source_id and normalize_str(cand.source_id) in eval_titles:
        return True
    t_norm = normalize_str(cand.title)
    c_norm = normalize_str(cand.composer)
    if t_norm and len(t_norm) >= 10:
        combo = f"{c_norm} {t_norm}".strip()
        if combo in eval_titles or t_norm in eval_slugs:
            return True
    return False


def parse_krn_headers(krn_path: Path) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    with krn_path.open("r", encoding="utf-8", errors="replace") as f:
        for _ in range(80):
            ln = f.readline()
            if not ln:
                break
            if ln.startswith("!!!"):
                if ":" in ln[3:]:
                    k, v = ln[3:].split(":", 1)
                    k = k.strip()
                    if k not in headers:
                        headers[k] = v.strip()
    return headers


def is_valid_ccby_yem(yem: str) -> bool:
    yl = (yem or "").lower()
    if not yl:
        return False
    if any(bad in yl for bad in ("non-commercial", "by-nc", "by-sa", "gpl", "[copyright holder]")):
        return False
    return (
        "creativecommons.org/licenses/by/4.0" in yl
        or "cc-by-4.0" in yl
        or "cc by 4.0" in yl
    )


def discover_candidates(repo_dir: Path) -> Tuple[List[Candidate], Dict[str, int]]:
    """Scan all .krn files and return ordered manuscript candidates."""
    stats = collections.Counter()
    sbb_cands: List[Candidate] = []
    nifc_by_ms: Dict[str, List[Candidate]] = collections.OrderedDict()

    krns = sorted(repo_dir.rglob("*.krn"))
    stats["total_krn_files"] = len(krns)

    for p in krns:
        rel = p.relative_to(repo_dir)
        arch = rel.parts[0]
        hdr = parse_krn_headers(p)
        yem = hdr.get("YEM", "")
        if not is_valid_ccby_yem(yem):
            stats["rejected_yem_license"] += 1
            continue

        url_scan_raw = hdr.get("URL-scan", "")
        url_scan = url_scan_raw.split()[0] if url_scan_raw else ""
        if not url_scan:
            stats["skipped_no_url_scan"] += 1
            continue

        composer = hdr.get("COM", "")
        title = hdr.get("OTL", "") or hdr.get("OPR", "")
        shelf = hdr.get("SMS-shelfmark", "")
        work_id = slugify(p.stem, max_len=110)

        if arch == "d-b" and "digital.staatsbibliothek-berlin.de" in url_scan:
            m_ppn = re.search(r"PPN=(PPN[0-9A-Za-z]+)", url_scan)
            if m_ppn and any(k in shelf.lower() for k in ("ms", "mus.ms", "autogr")):
                m_phys = re.search(r"PHYSID=PHYS_(\d+)", url_scan)
                page_hint = int(m_phys.group(1)) if m_phys else 1
                sbb_cands.append(
                    Candidate(
                        work_id=work_id,
                        archive=arch,
                        rel_path=str(rel).replace("\\", "/"),
                        krn_path=str(p),
                        source_kind="sbb_pdm",
                        source_id=m_ppn.group(1),
                        url_scan=url_scan,
                        page_hint=page_hint,
                        composer=composer,
                        title=title,
                        shelfmark=shelf,
                        yem_license=yem,
                    )
                )
                stats["cand_sbb_pdm"] += 1
                continue

        if "polish.musicsources.pl" in url_scan:
            if "/rekopisy/" in url_scan:
                m_ms = re.search(r"/rekopisy/(\d+)(?:/(\d+))?", url_scan)
                if m_ms:
                    ms_id = m_ms.group(1)
                    page_hint = int(m_ms.group(2)) if m_ms.group(2) else 0
                    onb = hdr.get("ONB-nifc", "")
                    if page_hint <= 1 and onb:
                        m_sk = re.search(r"skan\s+0*(\d+)", onb, re.I)
                        if m_sk:
                            page_hint = int(m_sk.group(1))
                    c = Candidate(
                        work_id=work_id,
                        archive=arch,
                        rel_path=str(rel).replace("\\", "/"),
                        krn_path=str(p),
                        source_kind="nifc_popc2_pd",
                        source_id=ms_id,
                        url_scan=url_scan,
                        page_hint=page_hint,
                        composer=composer,
                        title=title,
                        shelfmark=shelf,
                        yem_license=yem,
                    )
                    nifc_by_ms.setdefault(ms_id, []).append(c)
                    stats["cand_nifc_rekopisy"] += 1
                    continue
            elif "/druki-muzyczne/" in url_scan:
                stats["excluded_print_druki"] += 1
                continue
            else:
                stats["excluded_musicsources_other"] += 1
                continue

        if "bc.bdsandomierz.pl" in url_scan:
            stats["skipped_sandomierz_djvu_only"] += 1
        elif "polona" in url_scan:
            stats["excluded_polona_prints"] += 1
        elif "gallica.bnf.fr" in url_scan:
            stats["blocked_gallica_nc"] += 1
        else:
            stats["skipped_other_portal"] += 1

    # Round-robin across NIFC archives, picking 1 work per unique manuscript first
    by_arch_queues: Dict[str, List[List[Candidate]]] = collections.OrderedDict()
    arch_order = ["pl-wtm", "pl-cz", "pl-wumfc", "pl-stab", "pl-kozmzk", "pl-kk", "pl-gd", "pl-wnifc"]
    for ms_id, group in nifc_by_ms.items():
        by_arch_queues.setdefault(group[0].archive, []).append(group)

    ordered_nifc: List[Candidate] = []
    arch_keys = [a for a in arch_order if a in by_arch_queues] + [
        a for a in by_arch_queues if a not in arch_order
    ]
    arch_idx = {a: 0 for a in arch_keys}
    leftover_groups: List[List[Candidate]] = []
    while True:
        progressed = False
        for a in arch_keys:
            q = by_arch_queues[a]
            i = arch_idx[a]
            if i < len(q):
                grp = q[i]
                arch_idx[a] = i + 1
                ordered_nifc.append(grp[0])
                if len(grp) > 1:
                    leftover_groups.append(grp[1:])
                progressed = True
        if not progressed:
            break
    max_rem = max((len(g) for g in leftover_groups), default=0)
    for k in range(max_rem):
        for g in leftover_groups:
            if k < len(g):
                ordered_nifc.append(g[k])

    # Interleave SBB gently (1 SBB every 5 NIFC works) so SBB is never hammered concurrently
    merged: List[Candidate] = []
    si = 0
    ni = 0
    while si < len(sbb_cands) or ni < len(ordered_nifc):
        if ni < len(ordered_nifc):
            take = min(4, len(ordered_nifc) - ni)
            merged.extend(ordered_nifc[ni : ni + take])
            ni += take
        if si < len(sbb_cands):
            merged.append(sbb_cands[si])
            si += 1

    stats["unique_nifc_manuscripts"] = len(nifc_by_ms)
    stats["total_candidates"] = len(merged)
    return merged, dict(stats)


def convert_symbolic_worker(args_tuple: Tuple[str, str, str, str, str]) -> Tuple[str, bool, Dict[str, Any]]:
    """Top-level ProcessPoolExecutor worker: **kern -> MusicXML -> post-I1 bekern -> bekern-decode rc=0."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    work_id, krn_path_s, xml_out_s, bekern_out_s, aser_bin_s = args_tuple
    krn_path = Path(krn_path_s)
    xml_out = Path(xml_out_s)
    bekern_out = Path(bekern_out_s)
    aser_bin = Path(aser_bin_s)

    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    import music21
    from music21.musicxml import m21ToXml
    import recanon_i1

    logging.getLogger("music21").setLevel(logging.ERROR)

    # Fast resume: if both xml and bekern already exist and decode cleanly, reuse!
    if xml_out.exists() and bekern_out.exists():
        try:
            with tempfile.TemporaryDirectory() as td:
                dec_json = Path(td) / "dec.json"
                multiset = recanon_i1.decode_note_multiset(bekern_out, aser_bin, dec_json)
                n_pitched = sum(recanon_i1.pitched_only(multiset).values())
                if n_pitched > 0:
                    xml_txt = xml_out.read_text(encoding="utf-8", errors="replace")
                    m_nums = re.findall(r'<measure[^>]*number="(\d+)"', xml_txt)
                    return work_id, True, {
                        "xml_mode": "cached",
                        "measures": max((int(x) for x in m_nums), default=1),
                        "parts": xml_txt.count("<part "),
                        "bekern_tokens": len(bekern_out.read_text(encoding="utf-8").split()),
                        "i1_recanon_changed": False,
                        "decoded_pitched_notes": n_pitched,
                        "bekern_decode_rc": 0,
                    }
        except Exception:
            pass

    krn_text = krn_path.read_text(encoding="utf-8", errors="replace")
    try:
        sc = music21.converter.parseData(krn_text, format="humdrum")
    except Exception as e:
        return work_id, False, {"error": f"music21_parse:{type(e).__name__}:{str(e)[:100]}"}

    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        tmp_xml = td_p / "score.musicxml"
        bk_dir = td_p / "bk"
        bk_dir.mkdir()
        dec_json = td_p / "decoded.json"

        xml_mode = "std"
        try:
            sc.write("musicxml", fp=str(tmp_xml))
        except Exception:
            xml_mode = "no_makenot"
            try:
                gex = m21ToXml.GeneralObjectExporter(sc)
                gex.makeNotation = False
                tmp_xml.write_bytes(gex.parse())
            except Exception as e2:
                return work_id, False, {"error": f"musicxml_export:{type(e2).__name__}:{str(e2)[:100]}"}

        xml_bytes = tmp_xml.read_bytes()
        xml_txt = xml_bytes.decode("utf-8", errors="replace")
        m_nums = re.findall(r'<measure[^>]*number="(\d+)"', xml_txt)
        max_m = max((int(x) for x in m_nums), default=1)
        if max_m < 1:
            return work_id, False, {"error": "zero_measures"}

        r1 = subprocess.run(
            [
                str(aser_bin),
                "bekern",
                "--input",
                str(tmp_xml),
                "--out",
                str(bk_dir),
                "--stem",
                "work",
                "--measures",
                f"1..{max_m}",
            ],
            capture_output=True,
            text=True,
            timeout=90,
        )
        raw_bk = bk_dir / "work-bekern.txt"
        if r1.returncode != 0 or not raw_bk.exists():
            return work_id, False, {"error": f"bekern_rc={r1.returncode}:{r1.stderr.strip()[:120]}", "xml_mode": xml_mode}

        bk_text = raw_bk.read_text(encoding="utf-8")
        canon_text, i1_changed = recanon_i1.canonicalize_bekern_text(bk_text)
        canon_tmp = td_p / "canon-bekern.txt"
        canon_tmp.write_text(canon_text, encoding="utf-8")

        try:
            multiset = recanon_i1.decode_note_multiset(canon_tmp, aser_bin, dec_json)
        except Exception as e3:
            return work_id, False, {"error": f"bekern_decode:{str(e3)[:120]}", "xml_mode": xml_mode}

        n_pitched = sum(recanon_i1.pitched_only(multiset).values())
        if n_pitched <= 0:
            return work_id, False, {"error": "zero_pitched_notes", "xml_mode": xml_mode}

        atomic_write_bytes(xml_out, xml_bytes)
        atomic_write_text(bekern_out, canon_text)
        return work_id, True, {
            "xml_mode": xml_mode,
            "measures": max_m,
            "parts": len(sc.parts),
            "bekern_tokens": len(canon_text.split()),
            "i1_recanon_changed": i1_changed,
            "decoded_pitched_notes": n_pitched,
            "bekern_decode_rc": 0,
        }


def is_likely_music_page_cpu(img: Image.Image) -> Tuple[bool, str]:
    """Fast CPU-only filter rejecting blank pages, solid dark covers, and tiny/corrupt scans."""
    w, h = img.size
    if w < 400 or h < 400:
        return False, "too_small"
    tw = 360
    th = max(60, int(h * (tw / float(w))))
    gray = np.asarray(img.convert("L").resize((tw, th), Image.Resampling.BILINEAR), dtype=np.float32)

    y0, y1 = int(th * 0.08), int(th * 0.92)
    x0, x1 = int(tw * 0.08), int(tw * 0.92)
    roi = gray[y0:y1, x0:x1]
    if roi.size < 1000:
        return False, "empty_roi"

    std_val = float(roi.std())
    mean_val = float(roi.mean())
    if std_val < 9.0:
        return False, f"low_contrast(std={std_val:.1f})"
    if mean_val < 38.0:
        return False, f"dark_cover(mean={mean_val:.1f})"

    paper = float(np.percentile(roi, 78))
    if paper < 65.0:
        return False, f"dark_paper(p78={paper:.1f})"
    ink_thresh = max(28.0, paper - 28.0)
    ink_frac = float((roi < ink_thresh).mean())
    if ink_frac < 0.008:
        return False, f"blank_page(ink={ink_frac:.3f})"
    if ink_frac > 0.75:
        return False, f"heavy_cover(ink={ink_frac:.3f})"

    dy = np.abs(np.diff(roi, axis=0))
    edge_score = float((dy > 12.0).mean())
    if edge_score < 0.012:
        return False, f"low_detail(edge={edge_score:.3f})"

    return True, "ok"


def resize_and_encode_jpeg(img: Image.Image, max_dim: int = 1800, quality: int = 92) -> Tuple[bytes, int, int]:
    rgb = img.convert("RGB")
    w, h = rgb.size
    if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        rgb = rgb.resize((nw, nh), Image.Resampling.LANCZOS)
        w, h = rgb.size
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue(), w, h


def http_get_retry(
    session: requests.Session,
    url: str,
    timeout: float = 20.0,
    retries: int = 2,
    sbb_lock: Optional[threading.Lock] = None,
) -> Optional[requests.Response]:
    is_sbb = "staatsbibliothek-berlin.de" in url
    for attempt in range(retries):
        try:
            if is_sbb and sbb_lock is not None:
                with sbb_lock:
                    time.sleep(0.45)
                    r = session.get(url, timeout=timeout)
            else:
                r = session.get(url, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code in (403, 404, 410):
                return r
            if r.status_code == 429:
                if is_sbb:
                    # Fail fast if SBB is rate-limiting right now rather than stalling the pool
                    time.sleep(1.5)
                    if attempt == retries - 1:
                        return r
                else:
                    time.sleep(1.5 * (attempt + 1))
                continue
            if r.status_code in (500, 502, 503, 504):
                time.sleep(1.0 * (attempt + 1))
                continue
            return r
        except requests.RequestException:
            time.sleep(0.8 * (attempt + 1))
    return None


class MetadataCache:
    """Thread-safe cache for SBB METS/IIIF manifests and NIFC manuscript API responses."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.sbb_lock = threading.Lock()
        self._sbb: Dict[str, Tuple[bool, str, str, List[str]]] = {}
        self._nifc: Dict[str, Tuple[bool, str, str, List[str]]] = {}
        self.sbb_429_streak = 0

    def get_sbb(self, session: requests.Session, ppn: str) -> Tuple[bool, str, str, List[str]]:
        with self._lock:
            if ppn in self._sbb:
                return self._sbb[ppn]

        r_mets = http_get_retry(session, SBB_METS_URL.format(ppn=ppn), timeout=20.0, sbb_lock=self.sbb_lock)
        if r_mets is None or r_mets.status_code != 200:
            res = (False, "sbb_mets_fetch_failed", "", [])
        else:
            mets_txt = r_mets.text
            rejected = next((p.pattern for p in REJECT_LICENSE_PATTERNS if p.search(mets_txt)), None)
            if rejected:
                res = (False, f"sbb_rejected:{rejected}", "", [])
            else:
                lic_uri = ""
                for pat in ACCEPT_PDM_PATTERNS:
                    m = pat.search(mets_txt)
                    if m:
                        lic_uri = "https://" + m.group(0).lstrip("https://").lstrip("http://")
                        break
                if not lic_uri:
                    res = (False, "sbb_no_pdm_uri", "", [])
                else:
                    r_iiif = http_get_retry(session, SBB_IIIF_URL.format(ppn=ppn), timeout=20.0, sbb_lock=self.sbb_lock)
                    if r_iiif is None or r_iiif.status_code != 200:
                        res = (False, "sbb_iiif_fetch_failed", lic_uri, [])
                    else:
                        try:
                            mf = r_iiif.json()
                            canvases = (mf.get("sequences") or [{}])[0].get("canvases") or []
                            urls: List[str] = []
                            for c in canvases:
                                imgs = c.get("images") or []
                                if not imgs:
                                    continue
                                rsrc = imgs[0].get("resource") or {}
                                svc = rsrc.get("service")
                                svc_id = ""
                                if isinstance(svc, dict):
                                    svc_id = str(svc.get("@id") or svc.get("id") or "")
                                elif isinstance(svc, list) and svc and isinstance(svc[0], dict):
                                    svc_id = str(svc[0].get("@id") or svc[0].get("id") or "")
                                if svc_id:
                                    urls.append(f"{svc_id.rstrip('/')}/full/1600,/0/default.jpg")
                                elif rsrc.get("@id"):
                                    urls.append(str(rsrc["@id"]))
                            res = (True, "PDM-1.0", lic_uri, urls)
                        except Exception as e:
                            res = (False, f"sbb_iiif_parse_err:{e}", lic_uri, [])

        with self._lock:
            self._sbb[ppn] = res
        return res

    def get_nifc(self, session: requests.Session, ms_id: str) -> Tuple[bool, str, str, List[str]]:
        with self._lock:
            if ms_id in self._nifc:
                return self._nifc[ms_id]

        api_url = NIFC_API_MS_URL.format(ms_id=ms_id)
        r = http_get_retry(session, api_url, timeout=20.0)
        if r is None or r.status_code != 200:
            res = (False, f"nifc_api_http_{r.status_code if r else 'err'}", api_url, [])
        else:
            try:
                data = r.json()
                rights_note = str(data.get("rights_note") or "")
                rejected = next((p.pattern for p in REJECT_LICENSE_PATTERNS if p.search(rights_note)), None)
                if rejected:
                    res = (False, f"nifc_rights_rejected:{rejected}", api_url, [])
                else:
                    gallery = data.get("gallery") or []
                    urls: List[str] = []
                    for item in gallery:
                        if not isinstance(item, dict):
                            continue
                        img_urls = item.get("img_urls") or {}
                        u = img_urls.get("medium_size") or img_urls.get("obj")
                        if isinstance(u, str) and u.startswith("http"):
                            urls.append(u)
                    if not urls:
                        res = (False, "nifc_empty_gallery", api_url, [])
                    else:
                        res = (
                            True,
                            "EU-POPC2-PD-Art14",
                            "https://polish.musicsources.pl (EU POPC 2.3.2 Open Access / EU Dir 2019/1024 Art. 14; user-approved)",
                            urls,
                        )
            except Exception as e:
                res = (False, f"nifc_api_parse_err:{e}", api_url, [])

        with self._lock:
            self._nifc[ms_id] = res
        return res


def download_pages_for_work(
    cand: Candidate,
    conv_info: Dict[str, Any],
    meta_cache: MetadataCache,
    out_dir: Path,
    unlabeled_dir: Optional[Path],
    max_pages: int,
    max_dim: int,
    jpeg_quality: int,
) -> Dict[str, Any]:
    """Download up to max_pages verified music pages for a work that already passed symbolic conversion."""
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    t0 = time.monotonic()

    if cand.source_kind == "sbb_pdm":
        if meta_cache.sbb_429_streak >= 3:
            return {
                "status": "skipped_sbb_429_cooldown",
                "work_id": cand.work_id,
                "archive": cand.archive,
                "reason": "sbb_rate_limited_429",
            }
        lic_ok, lic_norm, lic_proof, img_urls = meta_cache.get_sbb(session, cand.source_id)
    else:
        lic_ok, lic_norm, lic_proof, img_urls = meta_cache.get_nifc(session, cand.source_id)

    if not lic_ok or not img_urls:
        return {
            "status": "rejected_license_or_meta",
            "work_id": cand.work_id,
            "archive": cand.archive,
            "reason": lic_norm,
        }

    n_total = len(img_urls)
    if cand.page_hint >= 1 and cand.page_hint <= n_total:
        start_idx = cand.page_hint - 1
    elif n_total >= 6:
        start_idx = 1
    else:
        start_idx = 0

    pages_meta: List[Dict[str, Any]] = []
    skipped_pages = 0
    max_attempts = min(n_total - start_idx, max_pages * 2 + 4)

    for offset in range(max_attempts):
        if len(pages_meta) >= max_pages:
            break
        cidx = start_idx + offset
        if cidx >= n_total:
            break
        u = img_urls[cidx]
        r_img = http_get_retry(
            session,
            u,
            timeout=25.0,
            retries=2,
            sbb_lock=meta_cache.sbb_lock if cand.source_kind == "sbb_pdm" else None,
        )
        if r_img is None or r_img.status_code != 200 or not r_img.content:
            if cand.source_kind == "sbb_pdm" and r_img is not None and r_img.status_code == 429:
                meta_cache.sbb_429_streak += 1
                if meta_cache.sbb_429_streak >= 3:
                    break
            skipped_pages += 1
            continue
        if cand.source_kind == "sbb_pdm":
            meta_cache.sbb_429_streak = 0

        try:
            im = Image.open(io.BytesIO(r_img.content))
            im.load()
        except Exception:
            skipped_pages += 1
            continue

        ok_pg, _pg_reason = is_likely_music_page_cpu(im)
        if not ok_pg:
            skipped_pages += 1
            continue

        jpg_bytes, w, h = resize_and_encode_jpeg(im, max_dim=max_dim, quality=jpeg_quality)
        page_fname = f"{cand.work_id}_p{cidx + 1:03d}.jpg"
        page_path = out_dir / "pages" / page_fname
        atomic_write_bytes(page_path, jpg_bytes)

        hardlinked = False
        if unlabeled_dir is not None:
            hardlinked = hardlink_or_copy(page_path, unlabeled_dir / page_fname)

        pages_meta.append(
            {
                "file": f"pages/{page_fname}",
                "canvas_index_1based": cidx + 1,
                "width": w,
                "height": h,
                "orientation": "portrait" if h >= w else "landscape",
                "bytes": len(jpg_bytes),
                "url": u,
                "hardlinked_unlabeled": hardlinked,
            }
        )

    if not pages_meta:
        return {
            "status": "failed_no_music_pages",
            "work_id": cand.work_id,
            "archive": cand.archive,
            "reason": f"0/{max_attempts}_pages_passed",
        }

    return {
        "status": "ok",
        "work_id": cand.work_id,
        "archive": cand.archive,
        "krn_rel_path": cand.rel_path,
        "source_kind": cand.source_kind,
        "source_id": cand.source_id,
        "url_scan": cand.url_scan,
        "composer": cand.composer,
        "title": cand.title,
        "shelfmark": cand.shelfmark,
        "krn_license": "CC-BY-4.0",
        "scan_license": lic_norm,
        "scan_license_proof": lic_proof,
        "xml_file": f"xml/{cand.work_id}.musicxml",
        "bekern_file": f"bekern/{cand.work_id}-bekern.txt",
        "symbolic": conv_info,
        "total_manuscript_canvases": n_total,
        "start_canvas_1based": start_idx + 1,
        "pages_downloaded": len(pages_meta),
        "pages_skipped_blank_or_err": skipped_pages,
        "pages": pages_meta,
        "elapsed_s": round(time.monotonic() - t0, 2),
    }


def write_usage_md(out_dir: Path, summary: Dict[str, Any]) -> None:
    md = f"""# NIFC Polish Heritage Handwritten Manuscript Corpus (`D:\\tmp\\nifc-polish-hw\\`)

Generated by `tmp-train/mvd/harvest_nifc_polish_hw.py` (CPU-only, resumable, atomic writes).

## Provenance & Licenses
- **Symbolic transcriptions (`**kern` → MusicXML → post-I1 canonical `bekern`)**:
  - Source: `https://github.com/pl-wnifc/humdrum-polish-scores` (The Fryderyk Chopin Institute / NIFC).
  - License: **`CC-BY 4.0`** (`https://creativecommons.org/licenses/by/4.0`, verified per-file `!!!YEM:` + repo `LICENSE.txt`; commercial use explicitly permitted; 45 placeholder/empty-header files excluded).
- **Handwritten manuscript scans (`pages/*.jpg`, `<= 1800 px` RGB JPEG Q=92)**:
  1. **NIFC Polish Music Sources (`polish.musicsources.pl/.../rekopisy/`)**: 16th–19th c. Polish heritage handwritten manuscripts digitized under EU POPC 2.3.2 / DMP Open Access (EU Directive 2019/1024 Art. 14; explicitly approved by user as Option 2). Strictly filtered to `/rekopisy/` ONLY (excludes `/druki-muzyczne/`, `/klocki-introligatorskie/`, `gallica.bnf.fr`, and any item with `NC`/`InC`/restricted text in `rights_note`).
  2. **Staatsbibliothek zu Berlin (`d-b`)**: verified per-PPN **`PDM 1.0`** (`https://creativecommons.org/publicdomain/mark/1.0/`) in SBB METS XML (`PPN1049147219` / `Slg Bohn Ms.mus. 111`, `PPN73488298X` / `Mus.ms. 19600/1`).
- **Audited & excluded sources**:
  - `bc.bdsandomierz.pl` (`pl-sa`, 1,162 works / 273 editions): explicit per-item OAI-PMH `<dc:rights>Domena publiczna (Public domain)</dc:rights>` + `<dc:type>rękopis muzyczny</dc:type>`, but serves indirect multi-page DjVu (`image/x.djvu`) only with no JPEG/PNG/PDF endpoint and no local `ddjvu` decoder.
  - `polona2.pl` / `polona.pl` (568 works): printed editions (`druk muzyczny`), excluded.
  - `gallica.bnf.fr` (37 works): hard-blocked (NC).

## Verified Corpus Summary
- **Total manuscript candidates in repo**: `{summary.get("discovery", {}).get("total_candidates", 0)}` (`{summary.get("discovery", {}).get("cand_sbb_pdm", 0)}` SBB `d-b` + `{summary.get("discovery", {}).get("cand_nifc_rekopisy", 0)}` NIFC `/rekopisy/` across `{summary.get("discovery", {}).get("unique_nifc_manuscripts", 0)}` unique manuscripts)
- **Harvested verified works**: `{summary.get("harvested_works", 0)}`
- **Harvested verified pages**: `{summary.get("harvested_pages", 0)}` (`portrait`: `{summary.get("portrait_pages", 0)}`, `landscape`: `{summary.get("landscape_pages", 0)}`)
- **Symbolic `**kern → bekern-decode rc=0` pass rate**: `{summary.get("symbolic_pass_rate_pct", 0.0):.1f}%` (`{summary.get("symbolic_passed", 0)}/{summary.get("symbolic_attempted", 0)}`)
- **Eval overlap (`suite31` + `suite31_hw`)**: `{summary.get("eval_overlap_excluded", 0)}` (asserted `0` in manifest)
- **Hardlinked unlabeled pages**: `{summary.get("unlabeled_hardlinks", 0)}` → `{summary.get("unlabeled_dir", "")}`

## Layout
- `pages/<work_id>_p<NNN>.jpg` — resized (`<= 1800 px`) RGB JPEG Q=92 manuscript scans
- `xml/<work_id>.musicxml` — MusicXML converted from diplomatic `**kern`
- `bekern/<work_id>-bekern.txt` — post-`I1` canonical bekern (`assembly-serialize.exe bekern-decode` rc=0)
- `manifest.jsonl` — per-work metadata, license proof, symbolic stats, and page list
- `summary.json` — aggregate metrics

## Resuming / Scaling
```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\\.venv\\Scripts\\python.exe tmp-train/mvd/harvest_nifc_polish_hw.py --limit 5000 --max-pages-per-work 8 --workers 6 --resume
```
"""
    atomic_write_text(out_dir / "USAGE.md", md)


def load_existing_manifest(manifest_path: Path) -> Dict[str, Dict[str, Any]]:
    existing: Dict[str, Dict[str, Any]] = {}
    if not manifest_path.exists():
        return existing
    for ln in manifest_path.read_text(encoding="utf-8", errors="replace").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            rec = json.loads(ln)
            wid = rec.get("work_id")
            if wid:
                existing[wid] = rec
        except json.JSONDecodeError:
            continue
    return existing


def main() -> int:
    ap = argparse.ArgumentParser(description="NIFC Polish Heritage Handwritten Manuscript Harvester (CPU-only)")
    ap.add_argument("--repo-dir", type=Path, default=DEFAULT_REPO_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--unlabeled-dir", type=Path, default=DEFAULT_UNLABELED_DIR)
    ap.add_argument("--no-hardlink", action="store_true", help="Do not hardlink pages into unlabeled-10k-real")
    ap.add_argument("--assembly-bin", type=Path, default=DEFAULT_ASSEMBLY_BIN)
    ap.add_argument("--limit", type=int, default=150, help="Target number of successfully harvested works")
    ap.add_argument("--max-pages-per-work", type=int, default=8)
    ap.add_argument("--max-dim", type=int, default=1800)
    ap.add_argument("--jpeg-quality", type=int, default=92)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if not args.repo_dir.exists():
        print(f"ERROR: repo dir not found: {args.repo_dir}", file=sys.stderr)
        return 1
    if not args.assembly_bin.exists():
        print(f"ERROR: assembly-serialize.exe not found: {args.assembly_bin}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "pages").mkdir(exist_ok=True)
    (args.out_dir / "xml").mkdir(exist_ok=True)
    (args.out_dir / "bekern").mkdir(exist_ok=True)

    unlabeled_dir = None if args.no_hardlink else args.unlabeled_dir
    if unlabeled_dir is not None:
        unlabeled_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.monotonic()
    eval_slugs, eval_titles = load_eval_guard()
    print(f"[guard] Loaded {len(eval_slugs)} eval slugs and {len(eval_titles)} eval titles/IDs", flush=True)

    candidates, disc_stats = discover_candidates(args.repo_dir)
    print(
        f"[discover] {disc_stats['total_krn_files']} .krn -> "
        f"{disc_stats['total_candidates']} manuscript candidates "
        f"(SBB d-b={disc_stats.get('cand_sbb_pdm', 0)}, "
        f"NIFC /rekopisy/={disc_stats.get('cand_nifc_rekopisy', 0)} across "
        f"{disc_stats.get('unique_nifc_manuscripts', 0)} unique manuscripts)",
        flush=True,
    )

    manifest_path = args.out_dir / "manifest.jsonl"
    existing = load_existing_manifest(manifest_path) if args.resume else {}
    if not args.resume and manifest_path.exists():
        manifest_path.unlink()

    if unlabeled_dir is not None and existing:
        for rec in existing.values():
            for pg in rec.get("pages", []):
                src = args.out_dir / pg["file"]
                if src.exists():
                    hardlink_or_copy(src, unlabeled_dir / src.name)

    eval_excluded = 0
    todo: List[Candidate] = []
    for c in candidates:
        if is_eval_overlap(c, eval_slugs, eval_titles):
            eval_excluded += 1
            continue
        if c.work_id in existing:
            continue
        todo.append(c)

    print(
        f"[queue] existing={len(existing)}, target_limit={args.limit}, "
        f"eval_excluded={eval_excluded}, queued={len(todo)}",
        flush=True,
    )

    meta_cache = MetadataCache()
    harvested: Dict[str, Dict[str, Any]] = dict(existing)
    fail_counts = collections.Counter()
    sym_attempted = len(existing)
    sym_passed = len(existing)
    manifest_lock = threading.Lock()

    cursor = 0
    while len(harvested) < args.limit and cursor < len(todo):
        remaining = args.limit - len(harvested)
        wave_size = min(len(todo) - cursor, max(args.workers * 2, int(remaining * 1.2) + 2))
        wave = todo[cursor : cursor + wave_size]
        cursor += wave_size
        cand_by_id = {c.work_id: c for c in wave}

        # Stage A: GIL-free parallel symbolic conversion (**kern -> MusicXML -> post-I1 bekern)
        t_sym = time.monotonic()
        sym_jobs = [
            (
                c.work_id,
                c.krn_path,
                str(args.out_dir / "xml" / f"{c.work_id}.musicxml"),
                str(args.out_dir / "bekern" / f"{c.work_id}-bekern.txt"),
                str(args.assembly_bin),
            )
            for c in wave
        ]
        passed_for_dl: List[Tuple[Candidate, Dict[str, Any]]] = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, args.workers)) as pex:
            for wid, ok, info in pex.map(convert_symbolic_worker, sym_jobs):
                sym_attempted += 1
                if ok:
                    sym_passed += 1
                    passed_for_dl.append((cand_by_id[wid], info))
                else:
                    fail_counts[f"failed_symbolic:{info.get('error', '')[:35]}"] += 1

        print(
            f"[wave-sym] {len(passed_for_dl)}/{len(wave)} passed bekern-decode rc=0 "
            f"in {time.monotonic() - t_sym:.1f}s (cumul {sym_passed}/{sym_attempted})",
            flush=True,
        )

        # Stage B: Parallel scan downloads for works that passed symbolic conversion
        if not passed_for_dl:
            continue
        t_dl = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as tex:
            futs = [
                tex.submit(
                    download_pages_for_work,
                    c,
                    cinfo,
                    meta_cache,
                    args.out_dir,
                    unlabeled_dir,
                    args.max_pages_per_work,
                    args.max_dim,
                    args.jpeg_quality,
                )
                for c, cinfo in passed_for_dl[: remaining + args.workers]
            ]
            for fut in concurrent.futures.as_completed(futs):
                try:
                    res = fut.result()
                except Exception as e:
                    fail_counts["dl_worker_exception"] += 1
                    print(f"[err] dl worker: {e}", file=sys.stderr, flush=True)
                    continue
                if res.get("status") == "ok":
                    if len(harvested) < args.limit:
                        wid = res["work_id"]
                        harvested[wid] = res
                        with manifest_lock:
                            with manifest_path.open("a", encoding="utf-8") as mf:
                                mf.write(json.dumps(res, ensure_ascii=False) + "\n")
                        if len(harvested) % 20 == 0 or len(harvested) == args.limit:
                            print(
                                f"[progress] {len(harvested)}/{args.limit} works "
                                f"(elapsed={time.monotonic() - t_start:.1f}s)",
                                flush=True,
                            )
                else:
                    fail_counts[f"{res.get('status')}:{res.get('reason', '')[:35]}"] += 1
        print(
            f"[wave-dl] harvested={len(harvested)}/{args.limit} "
            f"(dl_wave={time.monotonic() - t_dl:.1f}s, total={time.monotonic() - t_start:.1f}s)",
            flush=True,
        )

    # Assert 0 eval overlap
    for wid in harvested:
        if normalize_str(wid) in eval_slugs:
            raise AssertionError(f"Eval leakage detected: {wid}")

    by_archive = collections.Counter()
    by_source = collections.Counter()
    portrait = 0
    landscape = 0
    total_pages = 0
    total_tokens = 0
    total_notes = 0
    unlabeled_links = 0

    for rec in harvested.values():
        by_archive[rec["archive"]] += 1
        by_source[rec["source_kind"]] += 1
        sym = rec.get("symbolic") or {}
        total_tokens += int(sym.get("bekern_tokens", 0))
        total_notes += int(sym.get("decoded_pitched_notes", 0))
        for pg in rec.get("pages", []):
            total_pages += 1
            if pg.get("orientation") == "portrait":
                portrait += 1
            else:
                landscape += 1
            if unlabeled_dir is not None and (unlabeled_dir / Path(pg["file"]).name).exists():
                unlabeled_links += 1

    summary = {
        "repo_dir": str(args.repo_dir),
        "out_dir": str(args.out_dir),
        "unlabeled_dir": str(unlabeled_dir) if unlabeled_dir else "",
        "discovery": disc_stats,
        "eval_overlap_excluded": eval_excluded,
        "eval_overlap_in_manifest": 0,
        "harvested_works": len(harvested),
        "harvested_pages": total_pages,
        "portrait_pages": portrait,
        "landscape_pages": landscape,
        "portrait_ratio": round(portrait / total_pages, 4) if total_pages else 0.0,
        "unlabeled_hardlinks": unlabeled_links,
        "by_source_kind": dict(by_source),
        "by_archive": dict(by_archive.most_common()),
        "symbolic_attempted": sym_attempted,
        "symbolic_passed": sym_passed,
        "symbolic_pass_rate_pct": round(100.0 * sym_passed / sym_attempted, 2) if sym_attempted else 0.0,
        "total_bekern_tokens": total_tokens,
        "total_decoded_pitched_notes": total_notes,
        "failure_breakdown_top10": dict(fail_counts.most_common(10)),
        "elapsed_s": round(time.monotonic() - t_start, 2),
    }
    atomic_write_text(args.out_dir / "summary.json", json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    write_usage_md(args.out_dir, summary)

    print("\n=== HARVEST COMPLETE ===", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
