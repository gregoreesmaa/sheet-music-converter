#!/usr/bin/env python3
"""International Unlabeled Handwritten Music Manuscript Harvester (CPU-only).

Harvests strictly commercial-safe (PDM 1.0 / CC0 / CC-BY 4.0) unlabeled real
handwritten music manuscript pages from international public archives outside
Germany/US/Poland/Austria into `D:\\tmp\\unlabeled-10k-real\\intl_hw\\`.

Supported archives:
  1. e-manuscripta.ch (Switzerland — ZB Zürich, UB Basel, ETH, Swiss NL, etc.)
     via documented OAI-PMH (`set=noten`) + IIIF Presentation/Image API v2.
     Per-item `<dc:rights>` verified for `PDM 1.0` / `CC0` / `CC-BY 4.0`.
  2. Alvin Portal (Sweden — Uppsala Düben, Musikverket, Lund, Gothenburg)
     via documented Cora REST API (never bypasses legacy web Anubis PoW gate).
  3. BNE Digital / BDH (Spain) — documented API check only (never bypasses
     HTTP 403 WAF / reproduction gate).

Strict rules:
  - CPU-only (`CUDA_VISIBLE_DEVICES=""` set before any torch import).
  - License gate: per-item verification of PDM 1.0 / CC0 / CC-BY 4.0 / publicdomain;
    immediate rejection of any nc, noncommercial, in-copyright, gallica, or restricted rights.
  - Music-page filter: skips cover/page 1, prefers interior pages (max 4 pages/work),
    verifies every page with CPU-only CLIP (`openai/clip-vit-base-patch32`, `p_music >= 0.65`).
  - Resolution & format: JPEG <= 1800 px max side, quality=92, atomic tmp+rename.
  - Eval leakage guard: asserts 0 overlap with `suite31_hw_manifest.json`,
    `suite31_manifest.json`, and `gate_manifest.json`.

Usage:
    .\\.venv\\Scripts\\python.exe tmp-train/mvd/harvest_intl_unlabeled_hw.py \\
        --limit 450 --max-pages-per-work 4 --workers 6 --resume \\
        --out D:\\tmp\\unlabeled-10k-real\\intl_hw
"""

from __future__ import annotations

import os

# Hard-disable CUDA/GPU before torch can ever be imported (CPU + network only)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import argparse
import concurrent.futures
import hashlib
import io
import json
import re
import sys
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from urllib.parse import urlparse

import requests
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from harvest_unlabeled_10k import (  # noqa: E402
    CLIP_MUSIC_THRESHOLD,
    is_eval_excluded,
    is_music_page,
    load_eval_slugs,
    normalize_text,
)

USER_AGENT = "sheet-music-converter-research/1.0 (PD-OMR-Intl-HW; CPU-only)"
DEFAULT_OUT_DIR = Path(r"D:\tmp\unlabeled-10k-real\intl_hw")
DEFAULT_SUITE31_HW_MANIFEST = (
    REPO_ROOT / "data-gen" / "score-corpus" / "suite31" / "suite31_hw_manifest.json"
)
DEFAULT_SUITE31_MANIFEST = (
    REPO_ROOT / "data-gen" / "score-corpus" / "suite31" / "suite31_manifest.json"
)
DEFAULT_GATE20_MANIFEST = (
    REPO_ROOT / "data-gen" / "score-corpus" / "gate20" / "gate_manifest.json"
)

# Extra suite31_hw (composer, title_token) pairs beyond harvest_unlabeled_10k.py
EXTRA_SUITE31_HW_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("debussy", "vetement du blesse"),
    ("debussy", "berceuse heroique"),
    ("debussy", "la plus que lente"),
    ("debussy", "lindaraja"),
    ("hensel", "morgengruss"),
    ("hensel", "im herbste"),
    ("mendelssohn", "herbstlied"),
    ("beethoven", "blumchen wunderhold"),
    ("cornelius", "botschaft"),
)

# Strict license verification regexes
REJECT_LICENSE_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"gallica", re.I),
    re.compile(r"\bby-nc\b", re.I),
    re.compile(r"\bnoc-nc\b", re.I),
    re.compile(r"\bnc\b", re.I),
    re.compile(r"non[- ]?commercial", re.I),
    re.compile(r"in[- ]?copyright", re.I),
    re.compile(r"rightsstatements\.org/vocab/inc/", re.I),
    re.compile(r"all rights reserved", re.I),
    re.compile(r"urheberrecht", re.I),
    re.compile(r"benutzungsbestimmungen", re.I),
    re.compile(r"restricted", re.I),
    re.compile(r"\bby-nd\b", re.I),
)

ACCEPT_LICENSE_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"creativecommons\.org/publicdomain/mark/1\.0", re.I),
    re.compile(r"public domain mark\s*1\.0", re.I),
    re.compile(r"\bpdm\s*1\.0\b", re.I),
    re.compile(r"rightsstatements\.org/vocab/pdm/", re.I),
    re.compile(r"creativecommons\.org/publicdomain/zero/1\.0", re.I),
    re.compile(r"\bcc0\b", re.I),
    re.compile(r"creativecommons\.org/licenses/by/4\.0", re.I),
    re.compile(r"\bcc[- ]?by[- ]?4\.0\b", re.I),
    re.compile(r"\bpublic\s*domain\b", re.I),
    re.compile(r"publicdomain", re.I),
)

# Human-readable labels for e-manuscripta.ch OAI setSpecs
EMANUSCRIPTA_INST_MAP: Dict[str, str] = {
    "emanuszuz": "ZB Zürich",
    "emanuszuzneb": "ZB Zürich",
    "emanuszuzcmi": "ZB Zürich",
    "emanusbau": "UB Basel",
    "emanusbau1": "UB Basel",
    "emanuszut": "ETH-Bibliothek",
    "zuttma": "ETH-Bibliothek",
    "zuthsa": "ETH-Bibliothek",
    "emanussnl": "Swiss National Library",
    "bes": "UB Bern",
    "zhb": "ZHB Luzern",
    "zbs": "ZB Solothurn",
    "sbs": "Stadtbibliothek Schaffhausen",
    "win": "Winterthurer Bibliotheken",
    "sbz": "Stadtbibliothek Zofingen",
}

_HOST_LOCKS: Dict[str, threading.Semaphore] = {}
_HOST_LOCKS_GUARD = threading.Lock()


def get_host_semaphore(url: str) -> threading.Semaphore:
    host = (urlparse(url).hostname or "default").lower()
    with _HOST_LOCKS_GUARD:
        sem = _HOST_LOCKS.get(host)
        if sem is None:
            sem = threading.Semaphore(4)
            _HOST_LOCKS[host] = sem
        return sem


def http_get(
    session: requests.Session,
    url: str,
    params: Any = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 25.0,
    retries: int = 3,
) -> Optional[requests.Response]:
    sem = get_host_semaphore(url)
    for attempt in range(retries):
        try:
            with sem:
                r = session.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code in (400, 401, 403, 404, 410, 451):
                r.close()
                return None
            if r.status_code == 429:
                r.close()
                time.sleep(min(10.0, 2.5 * (attempt + 1)))
                continue
            r.close()
            time.sleep(1.0 * (attempt + 1))
        except requests.RequestException:
            time.sleep(1.0 * (attempt + 1))
    return None


def verify_license_text(rights_strings: Sequence[str]) -> Tuple[bool, str]:
    """Strictly verify PDM 1.0 / CC0 / CC-BY 4.0 / publicdomain; reject any NC/InC/restricted."""
    joined = " | ".join(s.strip() for s in rights_strings if s and s.strip())
    if not joined:
        return False, "no_license_metadata"
    for pat in REJECT_LICENSE_PATTERNS:
        m = pat.search(joined)
        if m:
            return False, f"rejected_license:{m.group(0)}"
    for pat in ACCEPT_LICENSE_PATTERNS:
        m = pat.search(joined)
        if m:
            return True, m.group(0)
    return False, f"unrecognized_license:{joined[:120]}"


def load_all_eval_slugs() -> Set[str]:
    """Load all eval slugs from gate20, suite31, and suite31_hw manifests."""
    slugs = load_eval_slugs(DEFAULT_GATE20_MANIFEST, DEFAULT_SUITE31_MANIFEST)
    if DEFAULT_SUITE31_HW_MANIFEST.is_file():
        try:
            data = json.loads(DEFAULT_SUITE31_HW_MANIFEST.read_text(encoding="utf-8"))
            for p in data.get("pieces", []):
                val = p.get("slug")
                if val:
                    norm = normalize_text(val)
                    if len(norm) >= 4:
                        slugs.add(norm)
                    if norm.startswith("hw "):
                        slugs.add(norm[3:].strip())
        except Exception:
            pass
    return slugs


def is_intl_eval_excluded(title: str, creator: str, work_id: str, eval_slugs: Set[str]) -> Tuple[bool, str]:
    combined = f"{creator} {title}"
    excluded, reason = is_eval_excluded(combined, item_id=work_id, extra_slugs=eval_slugs)
    if excluded:
        return True, reason
    norm = normalize_text(f"{work_id} {combined}")
    for comp, tok in EXTRA_SUITE31_HW_PAIRS:
        if comp in norm and re.search(rf"\b{re.escape(tok)}\b", norm):
            return True, f"suite31_hw_pair:{comp}+{tok}"
    return False, ""


def encode_jpeg_max_dim(img: Image.Image, max_dim: int = 1800, quality: int = 92) -> Tuple[bytes, int, int]:
    """Resize PIL image to <= max_dim on longest side and encode as RGB JPEG."""
    im = img.convert("RGB")
    w, h = im.size
    longest = max(w, h)
    if longest > max_dim:
        scale = max_dim / float(longest)
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        im = im.resize((nw, nh), Image.Resampling.LANCZOS)
        w, h = im.size
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue(), w, h


def discover_emanuscripta_works(
    oai_set: str = "noten",
    max_oai_pages: int = 80,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Harvest music manuscript metadata from e-manuscripta.ch OAI-PMH."""
    ns = {
        "o": "http://www.openarchives.org/OAI/2.0/",
        "dc": "http://purl.org/dc/elements/1.1/",
    }
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    works: List[Dict[str, Any]] = []
    stats = {"oai_pages": 0, "oai_records": 0, "license_rejected": 0, "printed_skipped": 0}
    token: Optional[str] = None

    try:
        for _ in range(max_oai_pages):
            if token:
                url = f"https://www.e-manuscripta.ch/oai/?verb=ListRecords&resumptionToken={token}"
            else:
                url = f"https://www.e-manuscripta.ch/oai/?verb=ListRecords&metadataPrefix=oai_dc&set={oai_set}"
            r = http_get(session, url, timeout=25.0)
            if r is None:
                break
            stats["oai_pages"] += 1
            try:
                root = ET.fromstring(r.text)
            except ET.ParseError:
                r.close()
                break
            r.close()

            for rec in root.findall(".//o:record", ns):
                header = rec.find("o:header", ns)
                if header is None or header.attrib.get("status") == "deleted":
                    continue
                stats["oai_records"] += 1
                ident = header.findtext("o:identifier", "", ns).strip()
                vlid = ident.split(":")[-1].strip()
                if not vlid.isdigit():
                    continue
                sets = [
                    s.text.strip()
                    for s in header.findall("o:setSpec", ns)
                    if s.text and s.text.strip() not in ("noten", "document")
                ]
                inst_code = sets[0] if sets else "e-manuscripta"
                inst_name = EMANUSCRIPTA_INST_MAP.get(inst_code, inst_code)

                titles = [t.text.strip() for t in rec.findall(".//dc:title", ns) if t.text and t.text.strip()]
                creators = [c.text.strip() for c in rec.findall(".//dc:creator", ns) if c.text and c.text.strip()]
                dates = [d.text.strip() for d in rec.findall(".//dc:date", ns) if d.text and d.text.strip()]
                types = [t.text.strip() for t in rec.findall(".//dc:type", ns) if t.text and t.text.strip()]
                rights = [rg.text.strip() for rg in rec.findall(".//dc:rights", ns) if rg.text and rg.text.strip()]

                # Skip explicitly printed items (e-manuscripta is overwhelmingly manuscripts,
                # but guard against occasional printed editions)
                type_joined = " ".join(types).lower()
                if "druck" in type_joined and "handschrift" not in type_joined and "manuscript" not in type_joined:
                    stats["printed_skipped"] += 1
                    continue

                ok_lic, lic_ev = verify_license_text(rights)
                if not ok_lic:
                    stats["license_rejected"] += 1
                    continue

                works.append(
                    {
                        "archive": "e-manuscripta.ch",
                        "country": "Switzerland",
                        "institution": inst_name,
                        "institution_code": inst_code,
                        "work_id": f"emanus_{vlid}",
                        "vlid": vlid,
                        "title": titles[0] if titles else f"e-manuscripta {vlid}",
                        "creator": "; ".join(creators[:2]),
                        "date": dates[0] if dates else "",
                        "license": " | ".join(rights),
                        "license_evidence": lic_ev,
                        "manifest_url": f"https://www.e-manuscripta.ch/i3f/v20/{vlid}/manifest",
                    }
                )

            rt = root.find(".//o:resumptionToken", ns)
            token = rt.text.strip() if (rt is not None and rt.text and rt.text.strip()) else None
            if not token:
                break
    finally:
        session.close()

    # Interleave across institutions & spread evenly so we maximize scribe/archive diversity
    by_inst: Dict[str, List[Dict[str, Any]]] = {}
    for w in works:
        by_inst.setdefault(w["institution"], []).append(w)
    interleaved: List[Dict[str, Any]] = []
    buckets = [by_inst[k] for k in sorted(by_inst.keys())]
    idx = 0
    while True:
        added = False
        for b in buckets:
            if idx < len(b):
                interleaved.append(b[idx])
                added = True
        if not added:
            break
        idx += 1
    return interleaved, stats


def probe_blocked_archives() -> Dict[str, Dict[str, Any]]:
    """Verify current API accessibility of BNE (Spain) and Alvin (Sweden) without bypassing gates."""
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    report: Dict[str, Dict[str, Any]] = {}
    try:
        # BNE Digital check
        try:
            r = session.get("https://bdh-rd.bne.es/oai?verb=Identify", timeout=10)
            report["bne_spain"] = {
                "status": "blocked_403_waf" if r.status_code == 403 else f"http_{r.status_code}",
                "http_code": r.status_code,
                "note": "BNE endpoints (*.bne.es) return HTTP 403 WAF/reproduction block; not bypassed.",
            }
            r.close()
        except Exception as e:
            report["bne_spain"] = {"status": "error", "note": str(e)}

        # Alvin Cora REST check for handwritten notated music
        try:
            sd = json.dumps(
                {
                    "name": "alvinRecordSearch",
                    "children": [
                        {
                            "name": "include",
                            "children": [
                                {
                                    "name": "includePart",
                                    "children": [
                                        {"name": "alvinRecordTypeSearchTerm", "value": "not"},
                                    ],
                                }
                            ],
                        },
                        {"name": "start", "value": "1"},
                        {"name": "rows", "value": "5"},
                    ],
                }
            )
            r = session.get(
                "https://cora.alvin-portal.org/rest/record/searchResult/alvinRecordSearch",
                params={"searchData": sd},
                headers={"Accept": "application/vnd.cora.recordList+json"},
                timeout=12,
            )
            total_not = int(r.json()["dataList"].get("totalNo", 0)) if r.status_code == 200 else -1
            r.close()
            report["alvin_sweden"] = {
                "status": "legacy_pow_gated_cora_empty_manuscripts",
                "cora_notated_music_total": total_not,
                "note": (
                    "Legacy www.alvin-portal.org is gated behind Anubis PoW anti-bot challenge "
                    "(not bypassed); open cora.alvin-portal.org REST API currently exposes only "
                    f"{total_not} notated-music record (print, 0 manuscripts; binarySearch=401)."
                ),
            }
        except Exception as e:
            report["alvin_sweden"] = {"status": "error", "note": str(e)}
    finally:
        session.close()
    return report


def extract_iiif_canvases(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract ordered (index, label, width, height, img_urls) from a IIIF v2 manifest."""
    seqs = manifest.get("sequences")
    if not isinstance(seqs, list) or not seqs:
        return []
    canvases = seqs[0].get("canvases")
    if not isinstance(canvases, list):
        return []
    out: List[Dict[str, Any]] = []
    for idx, c in enumerate(canvases):
        if not isinstance(c, dict):
            continue
        imgs = c.get("images")
        if not isinstance(imgs, list) or not imgs:
            continue
        res = imgs[0].get("resource")
        if not isinstance(res, dict):
            continue
        svc = res.get("service")
        svc_id = None
        if isinstance(svc, dict):
            svc_id = svc.get("@id") or svc.get("id")
        elif isinstance(svc, list) and svc and isinstance(svc[0], dict):
            svc_id = svc[0].get("@id") or svc[0].get("id")
        raw_id = res.get("@id") or res.get("id")

        urls: List[str] = []
        if svc_id:
            base = str(svc_id).rstrip("/")
            urls.append(f"{base}/full/!1800,1800/0/default.jpg")
            urls.append(f"{base}/full/1600,/0/default.jpg")
        if raw_id:
            urls.append(str(raw_id))
        if not urls:
            continue
        out.append(
            {
                "index": idx,
                "label": str(c.get("label", "")),
                "width": int(c.get("width") or 0),
                "height": int(c.get("height") or 0),
                "urls": urls,
            }
        )
    return out


def select_interior_canvases(canvases: List[Dict[str, Any]], max_candidates: int = 8) -> List[Dict[str, Any]]:
    """Skip cover/page 1 (index 0) and select interior pages spread across the manuscript."""
    n = len(canvases)
    if n <= 1:
        return []
    # Prefer interior pages: start at index 1 (or 2 if n >= 5), avoid very last back cover if n >= 4
    start = 2 if n >= 6 else 1
    end = n - 1 if n >= 5 else n
    pool = canvases[start:end]
    if not pool:
        pool = canvases[1:]
    if len(pool) <= max_candidates:
        return pool
    # Evenly spaced across the interior of the work
    idxs = [int(round(i * (len(pool) - 1) / float(max_candidates - 1))) for i in range(max_candidates)]
    seen: Set[int] = set()
    chosen: List[Dict[str, Any]] = []
    for i in idxs:
        if i not in seen:
            seen.add(i)
            chosen.append(pool[i])
    return chosen


class IntlHarvestState:
    def __init__(self, out_dir: Path, limit: int, resume: bool):
        self.out_dir = out_dir
        self.images_dir = out_dir / "images"
        self.manifest_path = out_dir / "manifest.jsonl"
        self.processed_path = out_dir / "processed_works.txt"
        self.summary_path = out_dir / "summary.json"
        self.limit = limit
        self.lock = threading.Lock()
        self.start_time = time.time()

        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.seen_hashes: Set[str] = set()
        self.processed_works: Set[str] = set()
        self.count = 0
        self.initial_count = 0
        self.by_archive: Dict[str, int] = {}
        self.by_institution: Dict[str, int] = {}
        self.by_license: Dict[str, int] = {}
        self.works_with_pages: Set[str] = set()
        self.portrait_count = 0
        self.landscape_count = 0

        # Counters
        self.works_inspected = 0
        self.eval_excluded = 0
        self.manifest_failed = 0
        self.pages_downloaded = 0
        self.pages_clip_passed = 0
        self.pages_clip_rejected = 0
        self.dup_skipped = 0
        self.clip_p_music_sum = 0.0

        if resume and self.manifest_path.is_file():
            for line in self.manifest_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                sha = rec.get("sha256")
                if sha:
                    self.seen_hashes.add(sha)
                self.count += 1
                arch = rec.get("archive", "unknown")
                inst = rec.get("institution", "unknown")
                lic = rec.get("license", "unknown")
                wid = rec.get("work_id", "")
                self.by_archive[arch] = self.by_archive.get(arch, 0) + 1
                self.by_institution[inst] = self.by_institution.get(inst, 0) + 1
                self.by_license[lic] = self.by_license.get(lic, 0) + 1
                if wid:
                    self.works_with_pages.add(wid)
                if rec.get("height", 0) >= rec.get("width", 0):
                    self.portrait_count += 1
                else:
                    self.landscape_count += 1
            self.initial_count = self.count
        elif not resume:
            if self.manifest_path.exists():
                self.manifest_path.unlink()
            if self.processed_path.exists():
                self.processed_path.unlink()

        if resume and self.processed_path.is_file():
            for line in self.processed_path.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if s:
                    self.processed_works.add(s)

    def is_done(self) -> bool:
        with self.lock:
            return self.count >= self.limit

    def mark_work_processed(self, work_id: str) -> None:
        with self.lock:
            if work_id in self.processed_works:
                return
            self.processed_works.add(work_id)
            with self.processed_path.open("a", encoding="utf-8") as f:
                f.write(work_id + "\n")
                f.flush()

    def save_page(
        self,
        work: Dict[str, Any],
        canvas_idx: int,
        image_url: str,
        jpg_bytes: bytes,
        width: int,
        height: int,
        clip_stats: Dict[str, Any],
    ) -> bool:
        sha = hashlib.sha256(jpg_bytes).hexdigest()
        with self.lock:
            if self.count >= self.limit:
                return False
            if sha in self.seen_hashes:
                self.dup_skipped += 1
                return False

            rec_id = f"{work['work_id']}_p{canvas_idx:03d}"
            fname = f"{rec_id}.jpg"
            final_path = self.images_dir / fname
            tmp_path = self.images_dir / (fname + ".tmp")
            tmp_path.write_bytes(jpg_bytes)
            os.replace(tmp_path, final_path)

            orient = "portrait" if height >= width else "landscape"
            rec = {
                "id": rec_id,
                "image_path": str(final_path),
                "source": work["archive"],
                "archive": work["archive"],
                "country": work["country"],
                "institution": work["institution"],
                "collection": "intl_hw",
                "work_id": work["work_id"],
                "item_id": work["work_id"],
                "title": work["title"],
                "creator": work["creator"],
                "date": work["date"],
                "license": work["license"],
                "license_verified": True,
                "license_evidence": work["license_evidence"],
                "manifest_url": work["manifest_url"],
                "url": image_url,
                "page_index": canvas_idx,
                "width": width,
                "height": height,
                "orientation": orient,
                "sha256": sha,
                "filter_stats": clip_stats,
                "page_class": clip_stats.get("page_class", "music_sheet"),
                "p_music": clip_stats.get("p_music", 0.0),
            }

            with self.manifest_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()

            self.seen_hashes.add(sha)
            self.count += 1
            self.by_archive[work["archive"]] = self.by_archive.get(work["archive"], 0) + 1
            self.by_institution[work["institution"]] = self.by_institution.get(work["institution"], 0) + 1
            self.by_license[work["license"]] = self.by_license.get(work["license"], 0) + 1
            self.works_with_pages.add(work["work_id"])
            if orient == "portrait":
                self.portrait_count += 1
            else:
                self.landscape_count += 1
            return True

    def write_summary(self, oai_stats: Dict[str, Any], blocked_report: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            elapsed = max(0.001, time.time() - self.start_time)
            tested = self.pages_clip_passed + self.pages_clip_rejected
            pass_rate = (self.pages_clip_passed / tested) if tested else 0.0
            summary = {
                "total_pages": self.count,
                "session_new_pages": max(0, self.count - self.initial_count),
                "distinct_works": len(self.works_with_pages),
                "limit": self.limit,
                "by_archive": dict(sorted(self.by_archive.items())),
                "by_institution": dict(sorted(self.by_institution.items())),
                "by_license": dict(sorted(self.by_license.items())),
                "portrait_count": self.portrait_count,
                "landscape_count": self.landscape_count,
                "portrait_frac": round(self.portrait_count / max(1, self.count), 4),
                "landscape_frac": round(self.landscape_count / max(1, self.count), 4),
                "works_inspected": self.works_inspected,
                "eval_excluded": self.eval_excluded,
                "manifest_failed": self.manifest_failed,
                "pages_downloaded": self.pages_downloaded,
                "pages_clip_passed": self.pages_clip_passed,
                "pages_clip_rejected": self.pages_clip_rejected,
                "clip_pass_rate": round(pass_rate, 4),
                "mean_p_music_passed": round(self.clip_p_music_sum / max(1, self.pages_clip_passed), 4),
                "dup_skipped": self.dup_skipped,
                "oai_discovery_stats": oai_stats,
                "other_archives_status": blocked_report,
                "elapsed_sec": round(elapsed, 2),
            }
            tmp = self.summary_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self.summary_path)
            return summary


def process_work(
    work: Dict[str, Any],
    state: IntlHarvestState,
    eval_slugs: Set[str],
    max_pages_per_work: int,
    max_dim: int,
    jpeg_quality: int,
    clip_threshold: float,
) -> int:
    if state.is_done():
        return 0
    wid = work["work_id"]
    with state.lock:
        if wid in state.processed_works:
            return 0
        state.works_inspected += 1

    # 1. Eval leakage guard
    excluded, _reason = is_intl_eval_excluded(work["title"], work["creator"], wid, eval_slugs)
    if excluded:
        with state.lock:
            state.eval_excluded += 1
        state.mark_work_processed(wid)
        return 0

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    saved_here = 0
    try:
        # 2. Fetch IIIF manifest
        r = http_get(session, work["manifest_url"], timeout=20.0)
        if r is None:
            with state.lock:
                state.manifest_failed += 1
            return 0
        try:
            manifest = r.json()
        except Exception:
            with state.lock:
                state.manifest_failed += 1
            return 0
        finally:
            r.close()

        # Defense-in-depth: also check manifest-level license/rights/attribution for any reject token
        extra_rights: List[str] = []
        for k in ("license", "rights", "attribution"):
            v = manifest.get(k)
            if isinstance(v, str) and v.strip():
                extra_rights.append(v.strip())
        if extra_rights:
            joined_extra = " | ".join(extra_rights)
            for pat in REJECT_LICENSE_PATTERNS:
                if pat.search(joined_extra):
                    return 0

        canvases = extract_iiif_canvases(manifest)
        candidates = select_interior_canvases(canvases, max_candidates=max(6, max_pages_per_work * 2))
        if not candidates:
            return 0

        for c in candidates:
            if state.is_done() or saved_here >= max_pages_per_work:
                break

            img: Optional[Image.Image] = None
            used_url = ""
            for u in c["urls"]:
                ri = http_get(session, u, timeout=25.0, retries=2)
                if ri is None:
                    continue
                try:
                    raw = ri.content
                finally:
                    ri.close()
                try:
                    im = Image.open(io.BytesIO(raw))
                    im.load()
                    img = im
                    used_url = u
                    break
                except Exception:
                    continue

            if img is None:
                continue

            with state.lock:
                state.pages_downloaded += 1

            ok, clip_stats = is_music_page(img, threshold=clip_threshold)
            if not ok:
                with state.lock:
                    state.pages_clip_rejected += 1
                continue

            with state.lock:
                state.pages_clip_passed += 1
                state.clip_p_music_sum += float(clip_stats.get("p_music", 0.0))

            jpg_bytes, w, h = encode_jpeg_max_dim(img, max_dim=max_dim, quality=jpeg_quality)
            if state.save_page(work, c["index"], used_url, jpg_bytes, w, h, clip_stats):
                saved_here += 1

        return saved_here
    finally:
        state.mark_work_processed(wid)
        session.close()


def verify_zero_eval_leakage(manifest_path: Path, eval_slugs: Set[str]) -> int:
    """Post-run assertion: verify 0 eval leakage across all harvested records."""
    leaks = 0
    if not manifest_path.is_file():
        return 0
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        ex, _ = is_intl_eval_excluded(
            rec.get("title", ""), rec.get("creator", ""), rec.get("work_id", ""), eval_slugs
        )
        if ex:
            leaks += 1
    return leaks


def main() -> int:
    ap = argparse.ArgumentParser(description="Harvest international PDM 1.0/CC0/CC-BY 4.0 HW music pages (CPU-only).")
    ap.add_argument("--limit", type=int, default=450, help="Target number of verified music pages.")
    ap.add_argument("--max-pages-per-work", type=int, default=4, help="Max pages per manuscript work.")
    ap.add_argument("--workers", type=int, default=6, help="Worker threads (CPU + network only).")
    ap.add_argument("--max-dim", type=int, default=1800, help="Max image side in pixels.")
    ap.add_argument("--jpeg-quality", type=int, default=92, help="JPEG quality.")
    ap.add_argument("--clip-threshold", type=float, default=CLIP_MUSIC_THRESHOLD, help="CLIP p_music threshold.")
    ap.add_argument("--max-oai-pages", type=int, default=80, help="Max OAI-PMH pages (10 works/page) from e-manuscripta.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help="Output directory.")
    ap.add_argument("--resume", action="store_true", help="Resume from existing manifest/processed list.")
    args = ap.parse_args()

    eval_slugs = load_all_eval_slugs()
    state = IntlHarvestState(args.out, limit=args.limit, resume=args.resume)

    print(f"[intl-hw] CPU-only harvester starting: target={args.limit}, already={state.count}, out={args.out}")
    blocked_report = probe_blocked_archives()
    for k, v in blocked_report.items():
        print(f"[intl-hw] archive probe {k}: {v.get('status')} — {v.get('note')}")

    if state.is_done():
        print(f"[intl-hw] Already at target ({state.count}/{args.limit}).")
        summary = state.write_summary({}, blocked_report)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    print(f"[intl-hw] Discovering PDM 1.0 music manuscripts from e-manuscripta.ch OAI-PMH (set=noten, up to {args.max_oai_pages} pages)...")
    works, oai_stats = discover_emanuscripta_works(oai_set="noten", max_oai_pages=args.max_oai_pages)
    print(f"[intl-hw] Discovered {len(works)} license-verified Swiss manuscripts (OAI stats: {oai_stats})")

    # Pre-warm CPU CLIP model once on main thread before spawning workers
    from harvest_unlabeled_10k import get_clip_classifier

    print("[intl-hw] Loading CPU CLIP classifier (openai/clip-vit-base-patch32)...")
    get_clip_classifier()
    print("[intl-hw] CLIP ready on CPU. Starting download + filter workers...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs: List[concurrent.futures.Future[int]] = []
        for w in works:
            if state.is_done():
                break
            futs.append(
                pool.submit(
                    process_work,
                    w,
                    state,
                    eval_slugs,
                    args.max_pages_per_work,
                    args.max_dim,
                    args.jpeg_quality,
                    args.clip_threshold,
                )
            )
        completed = 0
        for fut in concurrent.futures.as_completed(futs):
            completed += 1
            try:
                _ = fut.result()
            except Exception:
                pass
            if completed % 15 == 0 or state.is_done():
                summary = state.write_summary(oai_stats, blocked_report)
                print(
                    f"[intl-hw] progress: {summary['total_pages']}/{args.limit} pages "
                    f"({summary['distinct_works']} works, P/L={summary['portrait_count']}/{summary['landscape_count']}, "
                    f"CLIP pass={summary['clip_pass_rate']:.1%}, elapsed={summary['elapsed_sec']:.0f}s)",
                    flush=True,
                )
            if state.is_done():
                for f in futs:
                    f.cancel()
                break

    leaks = verify_zero_eval_leakage(state.manifest_path, eval_slugs)
    assert leaks == 0, f"Eval leakage detected: {leaks} records!"
    summary = state.write_summary(oai_stats, blocked_report)
    summary["eval_leakage_verified"] = leaks
    tmp = state.summary_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, state.summary_path)

    print("[intl-hw] DONE. Summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
