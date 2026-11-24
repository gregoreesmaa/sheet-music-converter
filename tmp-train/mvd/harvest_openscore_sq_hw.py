#!/usr/bin/env python3
"""OpenScore StringQuartets (CC0 1.0) x PDM 1.0 Autograph/Manuscript Harvester.

Cross-matches CC0 1.0 OpenScore StringQuartets against strictly verified PDM 1.0
handwritten manuscripts held at:
  - SBB Berlin (D-B, digital.staatsbibliothek-berlin.de)
  - SLUB Dresden (D-Dl, digital.slub-dresden.de)
  - BSB Munich (D-Mbs, digitale-sammlungen.de)

STRICT RULES:
  - CPU-only (`CUDA_VISIBLE_DEVICES=""` set before any imports).
  - 100% commercial-safe: PDM 1.0 scans x CC0 1.0 OpenScore StringQuartets ONLY
    (zero Gallica, zero NC, zero InC).
  - Zero eval leakage: excludes all suite31_hw landscape quartets, all real-hw-train
    Mozart quartets, all printed suite31 quartets, and all gate20 quartets.
  - Converts each matched `.mxl` -> `.musicxml` -> post-I1 canonical `bekern` via
    `assembly-serialize.exe bekern` + validates `bekern-decode` rc=0.
  - Uses CPU CLIP (`is_music_page`) to keep only verified music manuscript pages.
  - Atomic writes, <= 1800 px JPEG Q=92, resumable via `--resume`.

Usage:
    .\\.venv\\Scripts\\python.exe tmp-train/mvd/harvest_openscore_sq_hw.py \\
        --sq-dir D:\\tmp\\openscore-sq-repo\\scores \\
        --out D:\\tmp\\pdm-openscore-sq-hw \\
        --max-pages-per-work 12 --resume
"""

from __future__ import annotations

import os

# Hard-disable CUDA/GPU before anything else (CPU + network only)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import argparse
import concurrent.futures
import hashlib
import io
import json
import re
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import zipfile
from dataclasses import dataclass, field
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
)

USER_AGENT = "sheet-music-converter-research/1.0 (commercial-safe-pdm10-openscore-sq-harvester)"
DEFAULT_SQ_DIR = Path(r"D:\tmp\openscore-sq-repo\scores")
DEFAULT_OUT_DIR = Path(r"D:\tmp\pdm-openscore-sq-hw")
DEFAULT_ASSEMBLY_BIN = (
    REPO_ROOT / "converter" / "assembly-serialize" / "target" / "release" / "assembly-serialize.exe"
)

# Strictly allowed holding institutions (SBB Berlin, SLUB Dresden, BSB Munich ONLY)
ALLOWED_LIBRARIES: Dict[str, str] = {
    "D-B": "SBB Berlin",
    "D-Dl": "SLUB Dresden",
    "D-Mbs": "BSB Munich",
}

# Hard domain allowlist (PDM 1.0 German state libraries ONLY)
ALLOWED_HOST_SUFFIXES: Tuple[str, ...] = (
    "staatsbibliothek-berlin.de",
    "slub-dresden.de",
    "digitale-sammlungen.de",
)

# Hard blocklist substrings anywhere in URLs or rights metadata
FORBIDDEN_MARKERS: Tuple[str, ...] = (
    "gallica",
    "bnf.fr",
    "noc-nc",
    "non-commercial",
    "noncommercial",
    "by-nc",
    "c-nc",
    "by-nd",
    "rightsstatements.org/vocab/inc",
    "in copyright",
    "in-copyright",
    "all rights reserved",
    "urheberrechtsschutz",
)

# Strictly PDM 1.0 positive markers (must match at least one)
PDM10_MARKERS: Tuple[Tuple[str, str], ...] = (
    ("creativecommons.org/publicdomain/mark/1.0", "PDM 1.0"),
    ("public domain mark 1.0", "PDM 1.0"),
    ("pdm 1.0", "PDM 1.0"),
    ("rightsstatements.org/vocab/pdm/", "PDM 1.0"),
)

# Non-music canvas label keywords (bindings, covers, flyleaves, color charts)
NON_MUSIC_CANVAS_LABELS: Tuple[str, ...] = (
    "einband",
    "vorderdeckel",
    "ruckdeckel",
    "rueckdeckel",
    "deckel",
    "spiegel",
    "vorsatz",
    "nachsatz",
    "farbkeil",
    "colorchart",
    "color chart",
    "leerseite",
    "leer",
    "blank",
    "umschlag",
    "buchrucken",
    "buchruecken",
    "rucken",
    "ruecken",
    "schnitt",
    "binding",
    "cover",
    "spine",
    "flyleaf",
    "endpaper",
    "target",
    "titelblatt",
    "title page",
)

# STRICT EVAL & EXISTING-TRAIN EXCLUSIONS:
# Each entry: (composer_surname_norm, catalog_key_or_token, reason)
EXCLUDED_QUARTETS: Tuple[Tuple[str, str, str], ...] = (
    # 5 landscape quartets in suite31_hw_manifest.json
    ("beethoven", "op:135", "suite31_hw:hw-beethoven-op135"),
    ("mozart", "k:464", "suite31_hw:hw-mozart-k464"),
    ("mozart", "k:387", "suite31_hw:hw-mozart-k387"),
    ("mozart", "k:421", "suite31_hw:hw-mozart-k421"),
    ("mozart", "k:417b", "suite31_hw:hw-mozart-k421"),
    ("schubert", "d:810", "suite31_hw:hw-schubert-d810"),
    # 4 Mozart quartets already in D:\tmp\real-hw-train\
    ("mozart", "k:428", "real-hw-train:hw-mozart-k428"),
    ("mozart", "k:421b", "real-hw-train:hw-mozart-k428"),
    ("mozart", "k:458", "real-hw-train:hw-mozart-k458"),
    ("mozart", "k:465", "real-hw-train:hw-mozart-k465"),
    ("mozart", "k:590", "real-hw-train:hw-mozart-k590"),
    # 6 printed suite31 quartets
    ("mozart", "k:155", "suite31:mozart-k155"),
    ("mozart", "k:134a", "suite31:mozart-k155"),
    ("beethoven", "op:18/1", "suite31:beethoven-op18n1"),
    ("beethoven", "op:133", "suite31:beethoven-grosse-fuge"),
    ("haydn", "op:74/1", "suite31:haydn-op74n1"),
    ("haydn", "hob3:72", "suite31:haydn-op74n1"),
    ("schumann", "op:41/1", "suite31:schumann-op41n1"),
    # gate20 quartets (defense-in-depth)
    ("mayer", "op:14", "gate20:sq7070319"),
    ("saint", "op:1/3", "gate20:sq8438915"),
)

# Mendelssohn Op. -> MWV R cross-reference (RISM uses MWV R numbers)
MENDELSSOHN_OP_TO_MWV: Dict[str, str] = {
    "op:12": "mwv:r25",
    "op:13": "mwv:r22",
    "op:44/1": "mwv:r30",
    "op:44/2": "mwv:r26",
    "op:44/3": "mwv:r28",
    "op:80": "mwv:r37",
    "op:81": "mwv:r35",
}

# Per-host politeness semaphores to prevent 429s
_HOST_LOCKS: Dict[str, threading.Semaphore] = {}
_HOST_LOCKS_GUARD = threading.Lock()


def get_host_semaphore(url: str) -> threading.Semaphore:
    host = (urlparse(url).hostname or "default").lower()
    with _HOST_LOCKS_GUARD:
        sem = _HOST_LOCKS.get(host)
        if sem is None:
            limit = 2 if "staatsbibliothek-berlin.de" in host else 4
            sem = threading.Semaphore(limit)
            _HOST_LOCKS[host] = sem
        return sem


def http_get(
    session: requests.Session,
    url: str,
    params: Any = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 25.0,
    retries: int = 3,
    allow_redirects: bool = True,
) -> Optional[requests.Response]:
    sem = get_host_semaphore(url)
    for attempt in range(retries):
        try:
            with sem:
                r = session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=timeout,
                    allow_redirects=allow_redirects,
                )
            if r.status_code == 200:
                return r
            if r.status_code in (400, 401, 403, 404, 410, 451):
                r.close()
                return None
            if r.status_code == 429:
                r.close()
                time.sleep(min(12.0, 2.5 * (attempt + 1)))
                continue
            r.close()
            time.sleep(1.0 * (attempt + 1))
        except requests.RequestException:
            time.sleep(1.0 * (attempt + 1))
    return None


def normalize_text(text: str) -> str:
    s = (text or "").replace("ß", "ss").replace("ẞ", "ss")
    s = s.replace("♭", " flat ").replace("♯", " sharp ")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return s


def slugify(text: str, max_len: int = 80) -> str:
    s = normalize_text(text).replace(" ", "-")
    s = re.sub(r"-+", "-", s).strip("-")
    return s[:max_len]


def is_allowed_url(url: str) -> bool:
    if not url or not url.startswith(("http://", "https://")):
        return False
    low = url.lower()
    for bad in FORBIDDEN_MARKERS:
        if bad in low:
            return False
    try:
        host = (urlparse(low).hostname or "").strip(".")
    except Exception:
        return False
    return any(host == suf or host.endswith("." + suf) for suf in ALLOWED_HOST_SUFFIXES)


def _flatten_strings(val: Any) -> List[str]:
    out: List[str] = []
    if val is None:
        return out
    if isinstance(val, str):
        s = val.strip()
        if s:
            out.append(s)
    elif isinstance(val, list):
        for item in val:
            out.extend(_flatten_strings(item))
    elif isinstance(val, dict):
        for v in val.values():
            out.extend(_flatten_strings(v))
    return out


def verify_iiif_pdm10(manifest: Dict[str, Any], manifest_url: str) -> Tuple[bool, str, str]:
    """Strictly verify that an IIIF manifest declares explicit PDM 1.0 rights.

    Returns `(is_safe, canonical_rights_or_reason, raw_evidence_snippet)`.
    """
    if not is_allowed_url(manifest_url):
        return False, "forbidden_host", manifest_url

    chunks: List[str] = []
    for key in ("rights", "license", "attribution", "requiredStatement"):
        if key in manifest:
            chunks.extend(_flatten_strings(manifest[key]))

    seqs = manifest.get("sequences")
    if isinstance(seqs, list) and seqs and isinstance(seqs[0], dict):
        for key in ("rights", "license", "attribution"):
            if key in seqs[0]:
                chunks.extend(_flatten_strings(seqs[0][key]))

    for item in manifest.get("metadata", []) or []:
        if isinstance(item, dict):
            lbl = " ".join(_flatten_strings(item.get("label"))).lower()
            if any(k in lbl for k in ("lizenz", "licen", "recht", "right", "copyright", "nutzung")):
                chunks.extend(_flatten_strings(item.get("value")))

    joined = " | ".join(chunks)
    low = joined.lower()
    if not low:
        return False, "no_license_metadata", ""

    for bad in FORBIDDEN_MARKERS:
        if bad in low:
            return False, f"rejected_marker:{bad}", joined[:200]

    for marker, canonical in PDM10_MARKERS:
        if marker in low:
            return True, canonical, marker

    return False, "no_verified_pdm10_marker", joined[:200]


def extract_catalog_keys(text: str, surname: str = "") -> Set[str]:
    """Extract canonical catalog keys like 'op:18/2', 'op:131', 'k:575', 'hob3:77', 'd:804', 'mwv:r30', 'g:213'."""
    keys: Set[str] = set()
    s = (text or "").replace("_", " ")

    # 1. Opus with sub-number: Op. 18 No. 2, op.18/2, op. 44, no. 1, KinB 18.2, McCB 51,2, McCS 41/2, GötS 4/1
    for m in re.finditer(
        r"\b(?:op\.?|kinb|mccb|mccs|g[oö]ts)\.?\s*(\d+)\s*(?:no\.?|nr\.?|/|[,.\s]+no\.?|,|\.)\s*(\d+)\b",
        s,
        flags=re.I,
    ):
        op_num, sub_num = int(m.group(1)), int(m.group(2))
        keys.add(f"op:{op_num}/{sub_num}")

    # 2. Standalone Opus (only if not followed immediately by a sub-number, or as fallback for single-work opus)
    for m in re.finditer(r"\b(?:op\.?|kinb|mccb|mccs)\.?\s*(\d+)\b(?!\s*(?:no\.?|nr\.?|/|\.\d|,\s*\d))", s, flags=re.I):
        op_num = int(m.group(1))
        keys.add(f"op:{op_num}")

    # 3. Köchel (Mozart): K.575, KV 575, K.421417b, K.155134a, K.8073f, K.160159a, K. 134b
    for m in re.finditer(r"\b(?:kv|k)\.?\s*(\d{2,3}[a-z]?)(?:/\d+)?\b", s, flags=re.I):
        keys.add(f"k:{m.group(1).lower()}")
    # Handle OpenScore concatenated K numbers like K.421417b, K.155134a, K.8073f, K.428421b, K.160159a
    for m in re.finditer(r"\bk\.?\s*(\d{2,3})(\d{2,3}[a-z])\b", s, flags=re.I):
        keys.add(f"k:{m.group(1).lower()}")
        keys.add(f"k:{m.group(2).lower()}")

    # 4. Hoboken III (Haydn string quartets): Hob.III32, Hob III:32, Hob. III:77, Hob. III1
    for m in re.finditer(r"\bhob\.?\s*iii\s*[:.]?\s*(\d+)\b", s, flags=re.I):
        keys.add(f"hob3:{int(m.group(1))}")

    # 5. Deutsch (Schubert): D.804, D 887, D-2 804
    for m in re.finditer(r"\bd(?:-2|\.)?\s*(\d{1,3}[a-z]?)\b", s, flags=re.I):
        # Avoid matching "D major" or "D minor" (those have letters not digits)
        keys.add(f"d:{m.group(1).lower()}")

    # 6. MWV R (Mendelssohn string quartets): MWV R 25, MWV R 30
    for m in re.finditer(r"\bmwv\s*r\s*(\d+)\b", s, flags=re.I):
        keys.add(f"mwv:r{int(m.group(1))}")

    # 7. Gérard (Boccherini): G.213, GerB 213
    for m in re.finditer(r"\b(?:gerb|g\.)\s*(\d+)\b", s, flags=re.I):
        keys.add(f"g:{int(m.group(1))}")

    # 8. Krebs (Dittersdorf): Kr.191
    for m in re.finditer(r"\bkr\.?\s*(\d+)\b", s, flags=re.I):
        keys.add(f"kr:{int(m.group(1))}")

    # 9. White (Viotti): WhiV 2a.2, W 2a.2
    for m in re.finditer(r"\b(?:whiv|w)\s*(\d[a-z]\.\d)\b", s, flags=re.I):
        keys.add(f"whiv:{m.group(1).lower()}")

    # 10. Mendelssohn Op -> MWV expansion
    if surname == "mendelssohn":
        for k in list(keys):
            if k in MENDELSSOHN_OP_TO_MWV:
                keys.add(MENDELSSOHN_OP_TO_MWV[k])

    # 11. Cherubini Quartet No. 1 (E-flat major, Autograph D-B Mus.ms.autogr. Cherubini, L. 166)
    if surname == "cherubini":
        low_s = s.lower()
        if "no.1" in low_s or "no 1" in low_s or "cherubini, l. 166" in low_s:
            keys.add("cherubini:q1")

    return keys


def check_excluded(surname: str, cat_keys: Set[str], text_norm: str) -> Optional[str]:
    """Return exclusion reason if work/record overlaps eval suites or real-hw-train."""
    for exc_comp, exc_key, reason in EXCLUDED_QUARTETS:
        if exc_comp in surname:
            if exc_key in cat_keys:
                return reason
    if "beethoven" in surname and ("grosse fuge" in text_norm or "grobe fuge" in text_norm):
        return "suite31:beethoven-grosse-fuge"
    if "schubert" in surname and ("death and the maiden" in text_norm or "tod und das madchen" in text_norm):
        return "suite31_hw:hw-schubert-d810"
    return None


@dataclass
class OpenScoreQuartet:
    composer_folder: str
    composer_display: str
    surname: str
    given: str
    work_folder: str
    work_display: str
    mxl_path: Path
    catalog_keys: Set[str] = field(default_factory=set)
    key_signature_text: str = ""
    excluded_reason: Optional[str] = None


def extract_tonality(text: str) -> str:
    """Extract normalized key like 'e flat major', 'c sharp minor', 'a minor'."""
    norm = normalize_text(text)
    m = re.search(r"\b([a-g])\s*(flat|sharp)?\s*(major|minor)\b", norm)
    if not m:
        return ""
    return " ".join(p for p in (m.group(1), m.group(2) or "", m.group(3)) if p)


def index_openscore_quartets(sq_dir: Path) -> Tuple[List[OpenScoreQuartet], int]:
    """Index all CC0 OpenScore StringQuartets works.

    Returns `(all_works, excluded_count)`.
    """
    works: List[OpenScoreQuartet] = []
    excluded_count = 0
    for cdir in sorted(sq_dir.iterdir()):
        if not cdir.is_dir():
            continue
        folder = cdir.name
        composer_display = folder.replace(",_", ", ").replace("_", " ")
        parts = folder.split(",_", 1)
        surname_raw = parts[0].replace("_", " ")
        given_raw = parts[1].replace("_", " ") if len(parts) > 1 else ""
        # Handle "Hensel,_Fanny_(Mendelssohn)" -> surname="hensel", given="fanny"
        given_clean = re.sub(r"\([^)]*\)", "", given_raw).strip()
        surname = normalize_text(surname_raw).split()[0] if normalize_text(surname_raw) else ""
        given = normalize_text(given_clean).split()[0] if normalize_text(given_clean) else ""

        for wdir in sorted(cdir.iterdir()):
            if not wdir.is_dir():
                continue
            mxls = sorted(wdir.glob("*.mxl"))
            if not mxls:
                mxls = sorted(wdir.rglob("*.mxl"))
            if not mxls:
                continue
            # Prefer top-level combined .mxl if present
            mxl_path = mxls[0]
            work_display = wdir.name.replace("_", " ")
            cat_keys = extract_catalog_keys(work_display, surname=surname)
            tonality = extract_tonality(work_display)
            excl = check_excluded(surname, cat_keys, normalize_text(work_display))
            if excl:
                excluded_count += 1
            works.append(
                OpenScoreQuartet(
                    composer_folder=folder,
                    composer_display=composer_display,
                    surname=surname,
                    given=given,
                    work_folder=wdir.name,
                    work_display=work_display,
                    mxl_path=mxl_path,
                    catalog_keys=cat_keys,
                    key_signature_text=tonality,
                    excluded_reason=excl,
                )
            )
    return works, excluded_count


def rism_search_quartets(
    session: requests.Session,
    composer_query: str,
    lib_siglum: str,
    max_pages: int = 5,
) -> List[Dict[str, Any]]:
    """Search RISM Online for digitized quartet manuscripts for a composer in a library."""
    items: List[Dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        params = [
            ("q", f"Quartets {composer_query}"),
            ("mode", "sources"),
            ("rows", "100"),
            ("page", str(page)),
            ("fq", "has-digitization:true"),
            ("fq", "source-type:manuscript"),
            ("fq", f"holding-institution:{lib_siglum}"),
            ("fq", "hide-composite-volumes:true"),
        ]
        r = http_get(
            session,
            "https://rism.online/search",
            params=params,
            headers={"Accept": "application/ld+json", "User-Agent": USER_AGENT},
            timeout=25.0,
            retries=2,
        )
        if r is None:
            break
        try:
            data = r.json()
        except Exception:
            break
        finally:
            r.close()
        batch = data.get("items", []) or []
        if not batch:
            break
        items.extend(batch)
        total = int(data.get("totalItems", 0) or 0)
        if len(items) >= total or len(batch) < 100:
            break
    return items


def match_rism_to_quartet(
    rism_item: Dict[str, Any],
    quartets: Sequence[OpenScoreQuartet],
    surname: str,
    given: str,
) -> Optional[Tuple[OpenScoreQuartet, float, str]]:
    """Match a RISM digitized manuscript record against local CC0 OpenScore quartets."""
    summary = rism_item.get("summary") or {}
    comp_vals = ((summary.get("sourceComposer") or {}).get("value") or {}).get("none") or []
    rism_comp_norm = normalize_text(" ".join(comp_vals))

    # Strict composer check
    if surname not in rism_comp_norm:
        return None
    if given and len(given) >= 3 and given not in rism_comp_norm:
        # Allow initials or Mendelssohn-Bartholdy / Fanny Hensel
        if not (surname == "hensel" and "fanny" in rism_comp_norm):
            return None

    label_en = ((rism_item.get("label") or {}).get("en") or [""])[0]
    if not label_en:
        return None
    low_lbl = label_en.lower()

    # Must be a Quartet (not Entr'acte, Symphony, Trio, Arrangement, Sketch, Fragment, Excerpt)
    if "quartet" not in low_lbl and "streichquartett" not in low_lbl:
        return None
    for bad_tag in ("arrangement", "sketch", "fragment", "excerpt", "entr'acte", "piano", "klavier", "flute", "flöte", "guitar"):
        if bad_tag in low_lbl:
            return None

    rism_cats = extract_catalog_keys(label_en, surname=surname)
    rism_norm = normalize_text(label_en)

    # Hard check against eval / real-hw-train exclusions on RISM record itself
    if check_excluded(surname, rism_cats, rism_norm):
        return None

    rism_tonality = extract_tonality(label_en)

    best: Optional[Tuple[OpenScoreQuartet, float, str]] = None
    for q in quartets:
        if q.excluded_reason:
            continue

        # 1. Exact catalog key overlap (strongest & cleanest!)
        shared = q.catalog_keys & rism_cats
        if shared:
            # Prefer specific sub-opus / catalog keys over bare opus
            specific = [k for k in shared if "/" in k or not k.startswith("op:")]
            score = 1.2 if specific else 1.0
            if q.key_signature_text and rism_tonality:
                if q.key_signature_text == rism_tonality:
                    score += 0.15
                else:
                    # Tonality conflict -> likely a different work/arrangement
                    continue
            reason = f"catalog:{','.join(sorted(shared))}"
            if best is None or score > best[1]:
                best = (q, score, reason)

    return best


def resolve_iiif_manifest_url(
    session: requests.Session,
    rism_source_url: str,
    lib_siglum: str,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Fetch RISM source JSON-LD, verify it's a string quartet manuscript, and resolve IIIF manifest URL."""
    r = http_get(
        session,
        rism_source_url,
        headers={"Accept": "application/ld+json", "User-Agent": USER_AGENT},
        timeout=20.0,
        retries=2,
    )
    if r is None:
        return None
    try:
        src_json = r.json()
        txt = r.text
    except Exception:
        return None
    finally:
        r.close()

    # Check sourceType is Manuscript
    stype = str(((src_json.get("sourceTypes") or {}).get("sourceType") or {}).get("type", ""))
    if stype and "Manuscript" not in stype:
        return None

    ext_items = (src_json.get("externalResources") or {}).get("items") or []
    urls: List[str] = []
    if isinstance(ext_items, list):
        for item in ext_items:
            u = str(item.get("url", "")).strip()
            if u:
                urls.append(u)
    urls.extend(re.findall(r'https?://[^\s"<>\\]+', txt))
    # Deduplicate preserving order
    seen_u: Set[str] = set()
    uniq_urls: List[str] = []
    for u in urls:
        if u not in seen_u:
            seen_u.add(u)
            uniq_urls.append(u)

    # 1. Direct IIIF manifest URL
    for u in uniq_urls:
        if ("iiif" in u.lower() and "manifest" in u.lower()) and is_allowed_url(u):
            return u, src_json

    # 2. SBB Berlin (D-B)
    if lib_siglum == "D-B":
        m_ppn = re.search(r"PPN([0-9]{7,}[0-9Xx]?)", txt)
        if m_ppn:
            cand = f"https://content.staatsbibliothek-berlin.de/dc/PPN{m_ppn.group(1)}/manifest"
            if is_allowed_url(cand):
                return cand, src_json
        for u in uniq_urls:
            if "staatsbibliothek-berlin.de" in u.lower():
                rr = http_get(session, u, timeout=15.0, retries=2, allow_redirects=True)
                if rr is not None:
                    final_url = rr.url
                    rr.close()
                    m2 = re.search(r"PPN([0-9]{7,}[0-9Xx]?)", final_url)
                    if m2:
                        cand = f"https://content.staatsbibliothek-berlin.de/dc/PPN{m2.group(1)}/manifest"
                        if is_allowed_url(cand):
                            return cand, src_json

    # 3. SLUB Dresden (D-Dl)
    if lib_siglum == "D-Dl":
        for u in uniq_urls:
            m_slub = re.search(r"digital\.slub-dresden\.de/id([0-9X]+)", u)
            if m_slub:
                cand = f"https://iiif.slub-dresden.de/iiif/2/{m_slub.group(1)}/manifest.json"
                if is_allowed_url(cand):
                    return cand, src_json

    # 4. BSB Munich (D-Mbs)
    if lib_siglum == "D-Mbs":
        for u in uniq_urls:
            low = u.lower()
            if "bsb" in low and ("digitale-sammlungen.de" in low or "mdz-nbn-resolving.de" in low):
                m_bsb = re.search(r"(bsb\d{6,})", low)
                if m_bsb:
                    cand = f"https://api.digitale-sammlungen.de/iiif/presentation/v2/{m_bsb.group(1)}/manifest"
                    if is_allowed_url(cand):
                        return cand, src_json

    return None


def is_non_music_canvas(label: str) -> bool:
    nl = normalize_text(label)
    if not nl:
        return False
    return any(tok in nl for tok in NON_MUSIC_CANVAS_LABELS)


def extract_iiif_canvases(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract normalized canvas descriptors from IIIF Presentation v2 or v3."""
    out: List[Dict[str, Any]] = []

    seqs = manifest.get("sequences")
    if isinstance(seqs, list) and seqs and isinstance(seqs[0], dict):
        for idx, c in enumerate(seqs[0].get("canvases") or []):
            if not isinstance(c, dict):
                continue
            imgs = c.get("images") or []
            if not imgs or not isinstance(imgs[0], dict):
                continue
            res = imgs[0].get("resource") or {}
            svc = res.get("service")
            svc_id = ""
            if isinstance(svc, dict):
                svc_id = str(svc.get("@id") or svc.get("id") or "")
            elif isinstance(svc, list) and svc and isinstance(svc[0], dict):
                svc_id = str(svc[0].get("@id") or svc[0].get("id") or "")
            res_id = str(res.get("@id") or res.get("id") or "")
            if (svc_id and is_allowed_url(svc_id)) or (res_id and is_allowed_url(res_id)):
                out.append(
                    {
                        "index": idx,
                        "id": str(c.get("@id") or c.get("id") or ""),
                        "label": " ".join(_flatten_strings(c.get("label"))),
                        "width": int(c.get("width") or 0),
                        "height": int(c.get("height") or 0),
                        "service_id": svc_id,
                        "resource_id": res_id,
                    }
                )
        return out

    items = manifest.get("items")
    if isinstance(items, list):
        for idx, c in enumerate(items):
            if not isinstance(c, dict):
                continue
            try:
                body = c["items"][0]["items"][0]["body"]
                if isinstance(body, list):
                    body = body[0]
                res_id = str(body.get("id") or body.get("@id") or "")
                svcs = body.get("service") or []
                if isinstance(svcs, dict):
                    svcs = [svcs]
                svc_id = str(svcs[0].get("id") or svcs[0].get("@id") or "") if svcs and isinstance(svcs[0], dict) else ""
            except Exception:
                continue
            if (svc_id and is_allowed_url(svc_id)) or (res_id and is_allowed_url(res_id)):
                out.append(
                    {
                        "index": idx,
                        "id": str(c.get("id") or c.get("@id") or ""),
                        "label": " ".join(_flatten_strings(c.get("label"))),
                        "width": int(c.get("width") or 0),
                        "height": int(c.get("height") or 0),
                        "service_id": svc_id,
                        "resource_id": res_id,
                    }
                )
    return out


def candidate_image_urls(canvas: Dict[str, Any], max_dim: int = 1800) -> List[str]:
    urls: List[str] = []
    svc = (canvas.get("service_id") or "").rstrip("/")
    res = canvas.get("resource_id") or ""
    if svc and is_allowed_url(svc):
        low = svc.lower()
        if "staatsbibliothek-berlin.de" in low:
            # SBB returns 429 on non-full IIIF size transforms; prefer resource_id or /full/full/
            if res and is_allowed_url(res):
                urls.append(res)
            urls.append(f"{svc}/full/full/0/default.jpg")
        elif "slub-dresden.de" in low and ".tif.medium.jpg" in svc:
            large_svc = svc.replace(".tif.medium.jpg", ".tif.large.jpg")
            urls.append(f"{large_svc}/full/!{max_dim},{max_dim}/0/default.jpg")
            urls.append(f"{svc}/full/!{max_dim},{max_dim}/0/default.jpg")
        else:
            urls.append(f"{svc}/full/!{max_dim},{max_dim}/0/default.jpg")
            urls.append(f"{svc}/full/{max_dim},/0/default.jpg")
            urls.append(f"{svc}/full/full/0/default.jpg")
    if res and is_allowed_url(res) and res not in urls:
        urls.append(res)
    return urls


def fetch_and_filter_canvas(
    session: requests.Session,
    canvas: Dict[str, Any],
    max_dim: int = 1800,
    clip_threshold: float = CLIP_MUSIC_THRESHOLD,
) -> Optional[Tuple[Image.Image, str, Dict[str, Any]]]:
    """Fetch a canvas image and verify it is a music notation page via CPU CLIP."""
    for u in candidate_image_urls(canvas, max_dim=max_dim):
        r = http_get(session, u, headers={"User-Agent": USER_AGENT}, timeout=30.0, retries=2)
        if r is None:
            continue
        try:
            if len(r.content) < 4096:
                continue
            im = Image.open(io.BytesIO(r.content)).convert("RGB")
            im.load()
        except Exception:
            continue
        finally:
            r.close()

        ok, clip_stats = is_music_page(im, threshold=clip_threshold)
        if not ok:
            return None

        w, h = im.size
        if max(w, h) > max_dim:
            scale = max_dim / float(max(w, h))
            im = im.resize(
                (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                Image.Resampling.LANCZOS,
            )
        return im, u, clip_stats
    return None


def extract_mxl_to_musicxml(mxl_path: Path, out_xml_path: Path) -> Path:
    """Unzip a CC0 OpenScore `.mxl` into a clean `.musicxml` file atomically."""
    out_xml_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(mxl_path, "r") as zf:
        xml_member = None
        if "META-INF/container.xml" in zf.namelist():
            ct = zf.read("META-INF/container.xml").decode("utf-8", errors="ignore")
            m = re.search(r'full-path="([^"]+)"', ct)
            if m and m.group(1) in zf.namelist():
                xml_member = m.group(1)
        if not xml_member:
            cands = [
                n for n in zf.namelist()
                if n.lower().endswith((".musicxml", ".xml")) and "meta-inf" not in n.lower()
            ]
            if not cands:
                raise RuntimeError(f"No MusicXML inside {mxl_path}")
            xml_member = max(cands, key=lambda n: zf.getinfo(n).file_size)
        data = zf.read(xml_member)
    tmp = out_xml_path.with_suffix(".musicxml.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, out_xml_path)
    return out_xml_path


def convert_musicxml_to_bekern(
    xml_path: Path,
    bekern_dir: Path,
    slug: str,
    assembly_bin: Path,
) -> Tuple[bool, Optional[Path], int, str]:
    """Convert MusicXML -> post-I1 canonical `bekern` and verify `bekern-decode` rc=0."""
    bekern_dir.mkdir(parents=True, exist_ok=True)
    final_bk = bekern_dir / f"{slug}-bekern.txt"

    with tempfile.TemporaryDirectory(prefix="os_sq_bk_") as td:
        td_path = Path(td)
        r1 = subprocess.run(
            [
                str(assembly_bin),
                "bekern",
                "--input",
                str(xml_path),
                "--out",
                str(td_path),
                "--page",
                "1",
                "--stem",
                slug,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        tmp_bk = td_path / f"{slug}-bekern.txt"
        if r1.returncode != 0 or not tmp_bk.is_file():
            return False, None, 0, f"bekern_rc{r1.returncode}:{r1.stderr.strip()[:160]}"

        bk_text = tmp_bk.read_text(encoding="utf-8").strip()
        toks = bk_text.split()
        if len(toks) < 10:
            return False, None, len(toks), f"bekern_too_short:{len(toks)}"

        dec_out = td_path / "decoded.json"
        r2 = subprocess.run(
            [
                str(assembly_bin),
                "bekern-decode",
                "--input",
                str(tmp_bk),
                "--out",
                str(dec_out),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if r2.returncode != 0:
            return False, None, len(toks), f"bekern_decode_rc{r2.returncode}:{r2.stderr.strip()[:160]}"

        tmp_final = final_bk.with_suffix(".txt.tmp")
        tmp_final.write_text(bk_text + "\n", encoding="utf-8")
        os.replace(tmp_final, final_bk)
        return True, final_bk, len(toks), ""


def load_existing_manifest(manifest_path: Path) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    if manifest_path.is_file():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("slug"):
                    records[rec["slug"]] = rec
            except Exception:
                continue
    return records


def write_usage_md(out_dir: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# OpenScore StringQuartets (CC0 1.0) × PDM 1.0 Handwritten Manuscripts",
        "",
        "## License & Provenance",
        "- **Manuscript Images**: 100% programmatically verified **Public Domain Mark 1.0 (`PDM 1.0`)** from German state libraries:",
        "  - SBB Berlin (`D-B`, `staatsbibliothek-berlin.de`)",
        "  - SLUB Dresden (`D-Dl`, `slub-dresden.de`)",
        "  - BSB Munich (`D-Mbs`, `digitale-sammlungen.de`)",
        "- **Zero Gallica, zero NC, zero InC**: enforced by hard domain allowlist + blocklist + IIIF manifest rights verification (raw IIIF manifests archived in `manifests_iiif/`).",
        "- **Ground-Truth Scores**: 100% **CC0 1.0** from `OpenScore/StringQuartets`, extracted to `xml/{slug}.musicxml` and converted to canonical post-`I1` `bekern` in `bekern/{slug}-bekern.txt` (100% `assembly-serialize bekern-decode` rc=0).",
        "- **Eval Leakage**: `eval_overlap = 0` (strictly excludes all 5 `suite31_hw` landscape quartets, all 4 `D:\\tmp\\real-hw-train` Mozart quartets, all 6 printed `suite31` quartets, and all `gate20` quartets).",
        "",
        "## Dataset Summary",
        f"- **OpenScore StringQuartets indexed**: {summary['openscore_works_indexed']} works ({summary['openscore_works_excluded']} excluded for eval/train isolation)",
        f"- **PDM 1.0 manuscripts matched & downloaded**: {summary['works_matched']} works ({summary['portrait_works']} portrait, {summary['landscape_works']} landscape)",
        f"- **Total verified music pages**: {summary['total_pages']} pages ({summary['portrait_pages']} portrait, {summary['landscape_pages']} landscape)",
        f"- **bekern-decode pass rate**: {summary['bekern_decode_passed']}/{summary['works_matched']} ({summary['bekern_decode_pass_rate']*100:.1f}%)",
        f"- **By library**: `{json.dumps(summary['by_library'])}`",
        f"- **By rights**: `{json.dumps(summary['by_rights'])}`",
        f"- **Eval overlap**: `{summary['eval_overlap_count']}`",
        "",
        "## Directory Layout",
        "- `pages/{slug}/page_001.jpg` ... — CLIP-verified handwritten music pages (<= 1800 px, RGB JPEG Q=92)",
        "- `xml/{slug}.musicxml` — CC0 1.0 OpenScore MusicXML ground truth",
        "- `bekern/{slug}-bekern.txt` — post-`I1` canonical `bekern` tokens (`bekern-decode` rc=0 verified)",
        "- `manifests_iiif/{slug}.json` — archived IIIF manifests with `PDM 1.0` rights statements",
        "- `manifest.jsonl` — per-work metadata, provenance, rights evidence, page paths & dimensions",
        "- `summary.json` — aggregate counts and verification metrics",
        "",
        "## Regeneration / Resume",
        "```powershell",
        '$env:CUDA_VISIBLE_DEVICES=""; .\\.venv\\Scripts\\python.exe tmp-train/mvd/harvest_openscore_sq_hw.py --sq-dir D:\\tmp\\openscore-sq-repo\\scores --out D:\\tmp\\pdm-openscore-sq-hw --max-pages-per-work 12 --resume',
        "```",
        "",
    ]
    tmp = out_dir / "USAGE.md.tmp"
    tmp.write_text("\n".join(lines), encoding="utf-8")
    os.replace(tmp, out_dir / "USAGE.md")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Harvest strictly verified PDM 1.0 handwritten manuscripts matched to CC0 OpenScore StringQuartets."
    )
    ap.add_argument("--sq-dir", type=Path, default=DEFAULT_SQ_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--assembly-bin", type=Path, default=DEFAULT_ASSEMBLY_BIN)
    ap.add_argument("--max-works", type=int, default=60, help="Maximum matched quartets to download.")
    ap.add_argument("--max-pages-per-work", type=int, default=12, help="Max CLIP-verified music pages per matched quartet.")
    ap.add_argument("--max-canvases-scan", type=int, default=30, help="Max canvases to inspect per manifest to collect music pages.")
    ap.add_argument("--max-dim", type=int, default=1800, help="Max pixel dimension for downloaded JPEGs.")
    ap.add_argument("--libraries", nargs="+", default=["D-B", "D-Dl", "D-Mbs"], choices=list(ALLOWED_LIBRARIES.keys()))
    ap.add_argument("--clip-threshold", type=float, default=CLIP_MUSIC_THRESHOLD)
    ap.add_argument("--resume", action="store_true", help="Resume and skip slugs already in manifest.jsonl.")
    ap.add_argument("--dry-run", action="store_true", help="Match and verify PDM 1.0 licenses without downloading pages.")
    args = ap.parse_args()

    if not args.assembly_bin.is_file():
        print(f"ERROR: assembly-serialize binary not found at {args.assembly_bin}", file=sys.stderr)
        return 1

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"
    existing = load_existing_manifest(manifest_path) if args.resume else {}
    if not args.resume and manifest_path.is_file() and not args.dry_run:
        manifest_path.unlink()

    eval_slugs = load_eval_slugs()

    print(f"[1/4] Indexing CC0 OpenScore StringQuartets in {args.sq_dir} ...", flush=True)
    all_quartets, excluded_count = index_openscore_quartets(args.sq_dir)
    by_composer: Dict[str, List[OpenScoreQuartet]] = {}
    for q in all_quartets:
        by_composer.setdefault(q.composer_folder, []).append(q)
    eligible_count = sum(1 for q in all_quartets if not q.excluded_reason)
    print(
        f"      Indexed {len(all_quartets)} CC0 works across {len(by_composer)} composers "
        f"({excluded_count} excluded for eval/train isolation, {eligible_count} eligible).",
        flush=True,
    )

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    print(f"[2/4] Searching RISM Online digitized quartet manuscripts in {args.libraries} ...", flush=True)
    candidates: List[Dict[str, Any]] = []

    for cfolder, quartets in sorted(by_composer.items()):
        eligible = [q for q in quartets if not q.excluded_reason and q.catalog_keys]
        if not eligible:
            continue
        surname = eligible[0].surname
        given = eligible[0].given
        # Query RISM using composer surname (or full name for ambiguous surnames)
        q_comp = surname.capitalize()
        if surname in ("schumann", "strauss", "wolf", "mayer", "bach", "bridge"):
            q_comp = f"{given.capitalize()} {surname.capitalize()}"

        for lib in args.libraries:
            rism_items = rism_search_quartets(session, q_comp, lib, max_pages=5)
            matched_here = 0
            for it in rism_items:
                res = match_rism_to_quartet(it, eligible, surname, given)
                if not res:
                    continue
                q, score, reason = res
                label_en = ((it.get("label") or {}).get("en") or [""])[0]
                rism_url = str(it.get("id", ""))
                rism_id = rism_url.rstrip("/").split("/")[-1]
                # Defense-in-depth eval check via harvest_unlabeled_10k
                excl2, _ = is_eval_excluded(f"{q.composer_display} {label_en}", rism_id, extra_slugs=eval_slugs)
                if excl2:
                    continue

                mat_types = [
                    ((m.get("typeLabel") or {}).get("en") or [""])[0]
                    for m in ((it.get("flags") or {}).get("materialTypes") or [])
                ]
                is_autograph = any("autograph" in m.lower() for m in mat_types)
                candidates.append(
                    {
                        "quartet": q,
                        "score": score + (0.25 if is_autograph else 0.0),
                        "reason": reason,
                        "library_siglum": lib,
                        "library": ALLOWED_LIBRARIES[lib],
                        "rism_url": rism_url,
                        "rism_id": rism_id,
                        "rism_label": label_en,
                        "material_type": ", ".join(m for m in mat_types if m) or "Manuscript",
                        "is_autograph": is_autograph,
                    }
                )
                matched_here += 1
            if matched_here:
                print(
                    f"      {cfolder:30s} | {lib:5s} | rism={len(rism_items):3d} -> matched={matched_here}",
                    flush=True,
                )

    candidates.sort(key=lambda c: (c["is_autograph"], c["score"]), reverse=True)
    print(
        f"[3/4] Resolving IIIF manifests & strictly verifying PDM 1.0 rights on {len(candidates)} candidates ...",
        flush=True,
    )

    verified_works: List[Dict[str, Any]] = []
    seen_manifests: Set[str] = set()
    seen_work_lib: Set[Tuple[str, str, str]] = set()
    license_rejected = 0
    no_iiif = 0

    for cand in candidates:
        q: OpenScoreQuartet = cand["quartet"]
        wl_key = (q.composer_folder, q.work_folder, cand["library_siglum"])
        if wl_key in seen_work_lib:
            continue

        resolved = resolve_iiif_manifest_url(session, cand["rism_url"], cand["library_siglum"])
        if not resolved:
            no_iiif += 1
            continue
        mf_url, _src_json = resolved
        if mf_url in seen_manifests or not is_allowed_url(mf_url):
            continue

        rm = http_get(session, mf_url, headers={"User-Agent": USER_AGENT}, timeout=25.0, retries=2)
        if rm is None:
            no_iiif += 1
            continue
        try:
            mf_json = rm.json()
        except Exception:
            no_iiif += 1
            continue
        finally:
            rm.close()

        ok_rights, rights_str, evidence = verify_iiif_pdm10(mf_json, mf_url)
        if not ok_rights:
            license_rejected += 1
            print(f"      [SKIP LICENSE] {cand['library']} | {q.work_display} -> {rights_str}", flush=True)
            continue

        all_canvases = extract_iiif_canvases(mf_json)
        music_canvases = [c for c in all_canvases if not is_non_music_canvas(c["label"])]
        if not music_canvases:
            music_canvases = all_canvases
        if len(music_canvases) < 2:
            continue

        p_cnt = sum(1 for c in music_canvases if c["height"] > c["width"] > 0)
        l_cnt = sum(1 for c in music_canvases if c["width"] > c["height"] > 0)
        orient = "portrait" if p_cnt >= l_cnt else "landscape"

        slug = slugify(f"{q.surname}-{q.work_folder}-{cand['library_siglum'].lower()}-{cand['rism_id']}")
        seen_manifests.add(mf_url)
        seen_work_lib.add(wl_key)

        verified_works.append(
            {
                "slug": slug,
                "cand": cand,
                "mf_url": mf_url,
                "mf_json": mf_json,
                "rights": rights_str,
                "rights_evidence": evidence,
                "canvases": music_canvases,
                "total_canvases": len(music_canvases),
                "orientation": orient,
            }
        )
        print(
            f"      [VERIFIED PDM 1.0] {slug:55s} | {cand['library']:12s} | {len(music_canvases):3d}c | {orient:9s} | {cand['material_type']} | {cand['reason']}",
            flush=True,
        )
        if len(verified_works) >= args.max_works:
            break

    print(
        f"\n[4/4] Verified {len(verified_works)} PDM 1.0 quartet manuscripts "
        f"(license_rejected={license_rejected}, no_iiif={no_iiif}, dry_run={args.dry_run}).",
        flush=True,
    )

    if args.dry_run:
        p_w = sum(1 for w in verified_works if w["orientation"] == "portrait")
        l_w = sum(1 for w in verified_works if w["orientation"] == "landscape")
        print(f"DRY RUN summary: {len(verified_works)} works ({p_w} portrait, {l_w} landscape).")
        return 0

    downloaded_records: Dict[str, Dict[str, Any]] = dict(existing)
    bekern_passed = sum(1 for r in downloaded_records.values() if r.get("bekern_decode_rc") == 0)
    bekern_failed = 0

    for idx, w in enumerate(verified_works, 1):
        slug = w["slug"]
        if args.resume and slug in downloaded_records:
            print(f"  [{idx}/{len(verified_works)}] RESUME skip {slug}", flush=True)
            continue

        cand = w["cand"]
        q: OpenScoreQuartet = cand["quartet"]

        # 1. Extract CC0 MusicXML & convert to canonical post-I1 bekern (+ bekern-decode rc=0)
        xml_out = out_dir / "xml" / f"{slug}.musicxml"
        try:
            extract_mxl_to_musicxml(q.mxl_path, xml_out)
        except Exception as e:
            print(f"  [{idx}/{len(verified_works)}] SKIP MXL extract error for {slug}: {e}", flush=True)
            continue

        ok_bk, bk_path, bk_toks, bk_err = convert_musicxml_to_bekern(
            xml_out, out_dir / "bekern", slug, args.assembly_bin
        )
        if not ok_bk or bk_path is None:
            bekern_failed += 1
            print(f"  [{idx}/{len(verified_works)}] SKIP bekern failed for {slug}: {bk_err}", flush=True)
            continue

        # 2. Download up to max_pages_per_work CLIP-verified music pages (<= max_dim, JPEG Q=92, atomic)
        pages_dir = out_dir / "pages" / slug
        pages_dir.mkdir(parents=True, exist_ok=True)
        canvases = w["canvases"][: args.max_canvases_scan]

        page_paths: List[str] = []
        page_dims: List[List[int]] = []
        page_urls: List[str] = []
        page_shas: List[str] = []
        clip_scores: List[float] = []

        for c in canvases:
            if len(page_paths) >= args.max_pages_per_work:
                break
            res_img = fetch_and_filter_canvas(
                session, c, max_dim=args.max_dim, clip_threshold=args.clip_threshold
            )
            if res_img is None:
                continue
            im, img_url, clip_stats = res_img
            page_idx = len(page_paths) + 1
            out_jpg = pages_dir / f"page_{page_idx:03d}.jpg"
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=92)
            jpg_bytes = buf.getvalue()
            tmp_jpg = out_jpg.with_suffix(".jpg.tmp")
            tmp_jpg.write_bytes(jpg_bytes)
            os.replace(tmp_jpg, out_jpg)

            page_paths.append(str(out_jpg))
            page_dims.append([im.size[0], im.size[1]])
            page_urls.append(img_url)
            page_shas.append(hashlib.sha256(jpg_bytes).hexdigest())
            clip_scores.append(round(float(clip_stats.get("music_prob", 0.0)), 4))

        if not page_paths:
            print(f"  [{idx}/{len(verified_works)}] SKIP 0 CLIP-verified music pages for {slug}", flush=True)
            continue

        # 3. Archive verified IIIF manifest JSON for license audit trail
        mf_out = out_dir / "manifests_iiif" / f"{slug}.json"
        mf_out.parent.mkdir(parents=True, exist_ok=True)
        tmp_mf = mf_out.with_suffix(".json.tmp")
        tmp_mf.write_text(json.dumps(w["mf_json"], ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_mf, mf_out)

        n_port = sum(1 for dw, dh in page_dims if dh > dw)
        n_land = sum(1 for dw, dh in page_dims if dw >= dh)
        actual_orient = "portrait" if n_port >= n_land else "landscape"

        rec = {
            "slug": slug,
            "composer": q.composer_display,
            "work": q.work_display,
            "catalog_keys": sorted(q.catalog_keys),
            "match_reason": cand["reason"],
            "library_siglum": cand["library_siglum"],
            "library": cand["library"],
            "iiif_manifest": w["mf_url"],
            "iiif_manifest_local": str(mf_out),
            "rights": w["rights"],
            "rights_evidence": w["rights_evidence"],
            "material_type": cand["material_type"],
            "is_autograph": cand["is_autograph"],
            "rism_url": cand["rism_url"],
            "rism_id": cand["rism_id"],
            "rism_label": cand["rism_label"],
            "n_pages": len(page_paths),
            "n_portrait_pages": n_port,
            "n_landscape_pages": n_land,
            "orientation": actual_orient,
            "page_paths": page_paths,
            "page_dims": page_dims,
            "page_urls": page_urls,
            "page_sha256s": page_shas,
            "clip_music_probs": clip_scores,
            "openscore_mxl": str(q.mxl_path),
            "xml_path": str(xml_out),
            "xml_license": "CC0-1.0 (OpenScore StringQuartets)",
            "bekern_path": str(bk_path),
            "bekern_token_count": bk_toks,
            "bekern_decode_rc": 0,
            "eval_overlap": False,
        }
        downloaded_records[slug] = rec
        bekern_passed += 1
        with manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

        print(
            f"  [{idx}/{len(verified_works)}] SAVED {slug} | {len(page_paths)}p ({n_port}P/{n_land}L) | "
            f"{actual_orient} | {w['rights']} | {cand['library']} | bekern={bk_toks}t (rc=0)",
            flush=True,
        )

    recs = list(downloaded_records.values())
    total_pages = sum(r["n_pages"] for r in recs)
    total_port_pages = sum(r.get("n_portrait_pages", 0) for r in recs)
    total_land_pages = sum(r.get("n_landscape_pages", 0) for r in recs)
    by_lib: Dict[str, int] = {}
    by_rights: Dict[str, int] = {}
    by_material: Dict[str, int] = {}
    for r in recs:
        by_lib[r["library"]] = by_lib.get(r["library"], 0) + 1
        by_rights[r["rights"]] = by_rights.get(r["rights"], 0) + 1
        mt = "Autograph" if r.get("is_autograph") else "Manuscript copy"
        by_material[mt] = by_material.get(mt, 0) + 1

    eval_ov_count = sum(1 for r in recs if r.get("eval_overlap", False))
    bk_pass = sum(1 for r in recs if r.get("bekern_decode_rc") == 0)

    summary = {
        "openscore_works_indexed": len(all_quartets),
        "openscore_works_excluded": excluded_count,
        "openscore_works_eligible": eligible_count,
        "works_matched": len(recs),
        "portrait_works": sum(1 for r in recs if r["orientation"] == "portrait"),
        "landscape_works": sum(1 for r in recs if r["orientation"] == "landscape"),
        "total_pages": total_pages,
        "portrait_pages": total_port_pages,
        "landscape_pages": total_land_pages,
        "by_library": by_lib,
        "by_rights": by_rights,
        "by_material_type": by_material,
        "xml_license": "CC0-1.0 (OpenScore StringQuartets)",
        "bekern_decode_passed": bk_pass,
        "bekern_decode_failed": bekern_failed,
        "bekern_decode_pass_rate": round(bk_pass / len(recs), 4) if recs else 0.0,
        "license_rejected": license_rejected,
        "eval_overlap_count": eval_ov_count,
        "works": [
            {
                "slug": r["slug"],
                "composer": r["composer"],
                "work": r["work"],
                "library": r["library"],
                "rights": r["rights"],
                "material_type": r["material_type"],
                "n_pages": r["n_pages"],
                "orientation": r["orientation"],
                "bekern_token_count": r["bekern_token_count"],
                "bekern_decode_rc": r["bekern_decode_rc"],
            }
            for r in recs
        ],
    }
    tmp_sum = out_dir / "summary.json.tmp"
    tmp_sum.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_sum, out_dir / "summary.json")
    write_usage_md(out_dir, summary)

    print("\n=== HARVEST SUMMARY ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "works"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
