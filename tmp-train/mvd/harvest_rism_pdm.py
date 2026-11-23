#!/usr/bin/env python3
"""Route-A Harvester: RISM Online (CC-BY 4.0) x PDM 1.0 IIIF Handwritten Manuscripts.

Harvests real handwritten music manuscript pages paired with human-transcribed
RISM MARC21 031 Plaine & Easie (PAE) incipits converted to canonical post-I1
`bekern` tokens and validated via `assembly-serialize.exe bekern` + `bekern-decode`.

STRICT RULES:
  - CPU ONLY (`CUDA_VISIBLE_DEVICES=""` set before any torch import).
  - COMMERCIAL-SAFE LICENSES ONLY:
      * RISM catalog & PAE incipits: CC-BY 4.0.
      * IIIF scans MUST come from allowlisted PDM 1.0 / CC0 / US-PD libraries:
        SLUB Dresden (`D-Dl`), SBB Berlin (`D-B`), BSB Munich (`D-Mbs`),
        Polona (`PL-Wn`), Library of Congress (`US-Wc`), ONB Vienna (`A-Wn`).
      * NEVER downloads BnF Gallica (`gallica.bnf.fr`) or any NC / InC URL.
      * Every IIIF manifest is programmatically verified for PDM 1.0 / CC0 / PD.
  - Human-transcribed labels ONLY (RISM 031 PAE incipits).
  - Eval-leakage protected (excludes suite31_hw, gate20, suite31).
  - Resumable (`--resume`) with atomic file writes and JSONL manifest append.

Usage:
    .\\.venv\\Scripts\\python.exe tmp-train/mvd/harvest_rism_pdm.py \\
        --limit 200 --out D:\\tmp\\rism-pdm-hw --workers 6 --resume
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
import subprocess
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from urllib.parse import urlparse

import music21
import requests
from PIL import Image

# Reuse eval-leakage exclusion & CPU-only CLIP page classifier from harvest_unlabeled_10k
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from harvest_unlabeled_10k import (  # noqa: E402
    CLIP_MUSIC_THRESHOLD,
    is_eval_excluded,
    is_music_page,
    load_eval_slugs,
    resize_and_encode_jpeg,
)

USER_AGENT = "sheet-music-converter-research/1.0 (RISM-CC-BY-4.0 x PDM-1.0-IIIF; CPU-only)"
RISM_API_BASE = "https://rism.online"
RISM_HEADERS = {
    "Accept": "application/ld+json",
    "User-Agent": USER_AGENT,
}
DEFAULT_OUT_DIR = Path(r"D:\tmp\rism-pdm-hw")
DEFAULT_ASSEMBLY_BIN = (
    REPO_ROOT / "converter" / "assembly-serialize" / "target" / "release" / "assembly-serialize.exe"
)

MEI_NS = "http://www.music-encoding.org/ns/mei"
ET.register_namespace("", MEI_NS)

# Strictly commercial-safe PDM 1.0 / CC0 / US-PD IIIF domains ONLY
ALLOWED_DOMAIN_SUFFIXES: Tuple[str, ...] = (
    "slub-dresden.de",
    "staatsbibliothek-berlin.de",
    "digitale-sammlungen.de",
    "polona.pl",
    "loc.gov",
    "onb.ac.at",
)

# Explicit hard blocklist (defense-in-depth)
BLOCKED_DOMAIN_SUBSTRINGS: Tuple[str, ...] = (
    "gallica.bnf.fr",
    "bnf.fr",
)

# License verification regexes
REJECT_LICENSE_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"gallica\.bnf\.fr", re.I),
    re.compile(r"\bby-nc\b", re.I),
    re.compile(r"\bnoc-nc\b", re.I),
    re.compile(r"non[- ]?commercial", re.I),
    re.compile(r"in[- ]?copyright", re.I),
    re.compile(r"rightsstatements\.org/vocab/inc/", re.I),
    re.compile(r"all rights reserved", re.I),
    re.compile(r"urheberrechtsschutz", re.I),
    re.compile(r"\bby-nd\b", re.I),
)

ACCEPT_LICENSE_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"creativecommons\.org/publicdomain/mark/1\.0", re.I),
    re.compile(r"public domain mark\s*1\.0", re.I),
    re.compile(r"\bpdm\s*1\.0\b", re.I),
    re.compile(r"rightsstatements\.org/vocab/pdm/", re.I),
    re.compile(r"creativecommons\.org/publicdomain/zero/1\.0", re.I),
    re.compile(r"\bcc0\b", re.I),
    re.compile(r"rightsstatements\.org/vocab/noc-us/", re.I),
    re.compile(r"no copyright\s*-\s*united states", re.I),
    re.compile(r"\bpublic domain\b", re.I),
)

# Institution search configs: (siglum, filter_fq)
INSTITUTION_CONFIGS: Dict[str, str] = {
    "D-Dl": "has-iiif:true",
    "D-B": "has-digitization:true",
    "D-Mbs": "has-iiif:true",
    "PL-Wn": "has-digitization:true",
    "US-Wc": "has-digitization:true",
    "A-Wn": "has-digitization:true",
}

# Per-host politeness semaphores to prevent 429s
_HOST_LOCKS: Dict[str, threading.Semaphore] = {}
_HOST_LOCKS_GUARD = threading.Lock()


def get_host_semaphore(url: str) -> threading.Semaphore:
    host = (urlparse(url).hostname or "default").lower()
    with _HOST_LOCKS_GUARD:
        sem = _HOST_LOCKS.get(host)
        if sem is None:
            # SBB Berlin is sensitive to burst traffic; keep it gentle
            limit = 2 if "staatsbibliothek-berlin.de" in host else 4
            sem = threading.Semaphore(limit)
            _HOST_LOCKS[host] = sem
        return sem


def is_allowed_url(url: str) -> bool:
    if not url or not url.startswith(("http://", "https://")):
        return False
    low = url.lower()
    for blk in BLOCKED_DOMAIN_SUBSTRINGS:
        if blk in low:
            return False
    host = (urlparse(low).hostname or "").strip(".")
    return any(host == suf or host.endswith("." + suf) for suf in ALLOWED_DOMAIN_SUFFIXES)


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
                time.sleep(min(10.0, 2.5 * (attempt + 1)))
                continue
            r.close()
            time.sleep(1.0 * (attempt + 1))
        except requests.RequestException:
            time.sleep(1.0 * (attempt + 1))
    return None


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
        for k, v in val.items():
            if k in ("@value", "value", "en", "de", "none", "fr"):
                out.extend(_flatten_strings(v))
            elif isinstance(v, (str, list, dict)):
                out.extend(_flatten_strings(v))
    return out


def verify_iiif_license(manifest: Dict[str, Any], manifest_url: str) -> Tuple[bool, str]:
    """Strictly verify that an IIIF manifest carries PDM 1.0 / CC0 / US-PD rights."""
    if not is_allowed_url(manifest_url):
        return False, f"disallowed_manifest_host:{manifest_url}"

    candidates: List[str] = []
    for key in ("license", "rights", "attribution", "requiredStatement"):
        if key in manifest:
            candidates.extend(_flatten_strings(manifest[key]))

    seqs = manifest.get("sequences")
    if isinstance(seqs, list) and seqs and isinstance(seqs[0], dict):
        for key in ("license", "rights", "attribution"):
            if key in seqs[0]:
                candidates.extend(_flatten_strings(seqs[0][key]))

    meta = manifest.get("metadata")
    if isinstance(meta, list):
        for entry in meta:
            if not isinstance(entry, dict):
                continue
            lbl = " ".join(_flatten_strings(entry.get("label"))).lower()
            if any(k in lbl for k in ("license", "lizenz", "rights", "rechte", "copyright", "nutzung")):
                candidates.extend(_flatten_strings(entry.get("value")))

    joined = " | ".join(candidates)
    if not joined:
        return False, "no_license_metadata"

    # Hard reject NC / InC / ND / Gallica
    for pat in REJECT_LICENSE_PATTERNS:
        m = pat.search(joined)
        if m:
            return False, f"rejected_license_token:{m.group(0)}"

    # Require positive PDM 1.0 / CC0 / PD match
    for pat in ACCEPT_LICENSE_PATTERNS:
        m = pat.search(joined)
        if m:
            return True, m.group(0)

    return False, f"unrecognized_license:{joined[:160]}"


def extract_iiif_manifest_url(
    session: requests.Session,
    source_json: Dict[str, Any],
    siglum: str,
) -> Optional[str]:
    """Extract an allowlisted IIIF manifest URL from a RISM source record."""
    ext_items = source_json.get("externalResources", {}).get("items", [])
    if not isinstance(ext_items, list):
        return None

    # 1. Direct IIIF manifest links first
    for item in ext_items:
        url = str(item.get("url", "")).strip()
        rtype = str(item.get("resourceType", ""))
        if not url:
            continue
        if ("IIIFManifest" in rtype or "iiif" in url.lower() or "manifest" in url.lower()) and is_allowed_url(url):
            return url

    # 2. Institution-specific resolver -> IIIF manifest conversion
    for item in ext_items:
        url = str(item.get("url", "")).strip()
        if not url:
            continue
        low = url.lower()
        # SBB Berlin: resolver.staatsbibliothek-berlin.de -> PPN -> content.staatsbibliothek-berlin.de/dc/{PPN}/manifest
        if "staatsbibliothek-berlin.de" in low:
            m = re.search(r"PPN=?(PPN[A-Za-z0-9]+|[0-9]{7,}[0-9Xx]?)", url)
            if not m:
                r = http_get(session, url, timeout=15.0, retries=2, allow_redirects=True)
                if r is not None:
                    final_url = r.url
                    r.close()
                    m = re.search(r"PPN=?(PPN[A-Za-z0-9]+|[0-9]{7,}[0-9Xx]?)", final_url)
            if m:
                ppn = m.group(1)
                if not ppn.upper().startswith("PPN"):
                    ppn = "PPN" + ppn
                cand = f"https://content.staatsbibliothek-berlin.de/dc/{ppn}/manifest"
                if is_allowed_url(cand):
                    return cand

        # SLUB Dresden: digital.slub-dresden.de/id{digits} -> iiif.slub-dresden.de/iiif/2/{digits}/manifest.json
        if "slub-dresden.de" in low:
            m = re.search(r"/id(\d+)", url)
            if m:
                cand = f"https://iiif.slub-dresden.de/iiif/2/{m.group(1)}/manifest.json"
                if is_allowed_url(cand):
                    return cand

        # BSB Munich: bsb{digits} -> api.digitale-sammlungen.de/iiif/presentation/v2/bsb{digits}/manifest
        if "bsb" in low and ("digitale-sammlungen.de" in low or "mdz-nbn-resolving.de" in low):
            m = re.search(r"(bsb\d{6,})", low)
            if m:
                cand = f"https://api.digitale-sammlungen.de/iiif/presentation/v2/{m.group(1)}/manifest"
                if is_allowed_url(cand):
                    return cand

        # ONB Vienna: data.onb.ac.at/dtl/{id} -> iiif.onb.ac.at/presentation/ABO/+Z... (via redirect)
        if "onb.ac.at" in low and "/dtl/" in low:
            r = http_get(session, url, timeout=15.0, retries=2, allow_redirects=True)
            if r is not None:
                m = re.search(r"https?://iiif\.onb\.ac\.at/presentation/[^\s\"'<>]+/manifest", r.text)
                r.close()
                if m and is_allowed_url(m.group(0)):
                    return m.group(0)

    return None


def parse_pae_timesig(ts: str) -> Optional[Tuple[str, str, Optional[str]]]:
    """Parse RISM PAE timesig (e.g. 'c', 'c/', '3/4', '6/8') -> (count, unit, sym)."""
    if not ts:
        return None
    s = ts.strip().lower()
    if s == "c":
        return ("4", "4", "common")
    if s in ("c/", "c|", "¢"):
        return ("2", "2", "cut")
    m = re.match(r"^(\d+)/(\d+)$", s)
    if m:
        return (m.group(1), m.group(2), None)
    return None


def patch_mei_for_music21(mei_xml: str, pae_props: Dict[str, Any]) -> str:
    """Lift MEI 6.0 child <keySig>/<meterSig> onto <scoreDef> attributes for music21."""
    root = ET.fromstring(mei_xml)
    ns = {"m": MEI_NS}
    ts_parsed = parse_pae_timesig(str(pae_props.get("timesig", "")))

    for sd in root.findall(".//m:scoreDef", ns):
        ks = sd.find("m:keySig", ns)
        if ks is not None:
            sig = ks.attrib.get("sig")
            if sig and "key.sig" not in sd.attrib:
                sd.attrib["key.sig"] = sig
            mode = ks.attrib.get("mode")
            if mode and "key.mode" not in sd.attrib:
                sd.attrib["key.mode"] = mode
        ms = sd.find("m:meterSig", ns)
        if ms is not None:
            if "count" in ms.attrib and "meter.count" not in sd.attrib:
                sd.attrib["meter.count"] = ms.attrib["count"]
            if "unit" in ms.attrib and "meter.unit" not in sd.attrib:
                sd.attrib["meter.unit"] = ms.attrib["unit"]
            if "sym" in ms.attrib and "meter.sym" not in sd.attrib:
                sd.attrib["meter.sym"] = ms.attrib["sym"]
        elif ts_parsed and "meter.count" not in sd.attrib:
            sd.attrib["meter.count"] = ts_parsed[0]
            sd.attrib["meter.unit"] = ts_parsed[1]
            if ts_parsed[2]:
                sd.attrib["meter.sym"] = ts_parsed[2]

    return ET.tostring(root, encoding="unicode")


def convert_incipit_to_bekern(
    session: requests.Session,
    incipit_item: Dict[str, Any],
    assembly_bin: Path,
) -> Tuple[Optional[str], Dict[str, Any], str]:
    """Fetch RISM MEI for an incipit -> MusicXML -> assembly-serialize bekern + bekern-decode.

    Returns `(bekern_str_or_None, meta_dict, error_reason)`.
    """
    inc_id = str(incipit_item.get("id", "")).strip()
    props = incipit_item.get("properties") or {}
    if not inc_id or not props.get("notation"):
        return None, {}, "missing_pae_notation"

    mei_url = inc_id.rstrip("/") + "/mei"
    for enc in incipit_item.get("encodings") or []:
        if enc.get("format") == "application/mei+xml" and enc.get("url"):
            mei_url = str(enc["url"])
            break

    r_mei = http_get(session, mei_url, headers={"User-Agent": USER_AGENT}, timeout=20.0, retries=2)
    if r_mei is None:
        return None, {}, "mei_fetch_failed"
    mei_raw = r_mei.text
    r_mei.close()

    try:
        mei_patched = patch_mei_for_music21(mei_raw, props)
    except ET.ParseError as e:
        return None, {}, f"mei_xml_parse_err:{e}"

    try:
        score = music21.converter.parseData(mei_patched, format="mei")
    except Exception as e:
        return None, {}, f"music21_mei_err:{type(e).__name__}:{str(e)[:120]}"

    with tempfile.TemporaryDirectory(prefix="rism_bk_") as td:
        td_path = Path(td)
        mxml_path = td_path / "incipit.musicxml"
        try:
            score.write("musicxml", fp=str(mxml_path))
        except Exception as e:
            return None, {}, f"musicxml_write_err:{type(e).__name__}:{str(e)[:120]}"

        r1 = subprocess.run(
            [
                str(assembly_bin),
                "bekern",
                "--input",
                str(mxml_path),
                "--out",
                str(td_path),
                "--page",
                "1",
                "--stem",
                "incipit",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if r1.returncode != 0:
            return None, {}, f"bekern_rc{r1.returncode}:{r1.stderr.strip()[:140]}"

        bk_path = td_path / "incipit-bekern.txt"
        if not bk_path.is_file():
            return None, {}, "bekern_missing_output"
        bekern_txt = bk_path.read_text(encoding="utf-8").strip()
        toks = bekern_txt.split()
        if len(toks) < 6:
            return None, {}, f"bekern_too_short:{len(toks)}"

        dec_path = td_path / "decoded.json"
        r2 = subprocess.run(
            [
                str(assembly_bin),
                "bekern-decode",
                "--input",
                str(bk_path),
                "--out",
                str(dec_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if r2.returncode != 0:
            return None, {}, f"bekern_decode_rc{r2.returncode}:{r2.stderr.strip()[:140]}"

    meta = {
        "incipit_id": inc_id,
        "mei_url": mei_url,
        "pae_clef": props.get("clef", ""),
        "pae_keysig": props.get("keysig", ""),
        "pae_timesig": props.get("timesig", ""),
        "pae_notation": props.get("notation", ""),
        "bekern_token_count": len(toks),
    }
    return bekern_txt, meta, ""


def extract_canvases(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract normalized canvas descriptors from IIIF Presentation v2 or v3."""
    out: List[Dict[str, Any]] = []

    # IIIF v2
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
            thumb = c.get("thumbnail")
            thumb_url = ""
            if isinstance(thumb, str):
                thumb_url = thumb
            elif isinstance(thumb, dict):
                thumb_url = str(thumb.get("@id") or thumb.get("id") or "")
            out.append(
                {
                    "index": idx,
                    "label": " ".join(_flatten_strings(c.get("label"))),
                    "width": c.get("width") or 0,
                    "height": c.get("height") or 0,
                    "service_id": svc_id,
                    "resource_id": res_id,
                    "thumbnail_url": thumb_url,
                }
            )
        return out

    # IIIF v3
    items = manifest.get("items")
    if isinstance(items, list):
        for idx, c in enumerate(items):
            if not isinstance(c, dict):
                continue
            ann_pages = c.get("items") or []
            if not ann_pages or not isinstance(ann_pages[0], dict):
                continue
            anns = ann_pages[0].get("items") or []
            if not anns or not isinstance(anns[0], dict):
                continue
            body = anns[0].get("body") or {}
            if isinstance(body, list) and body:
                body = body[0]
            svc = body.get("service")
            svc_id = ""
            if isinstance(svc, list) and svc and isinstance(svc[0], dict):
                svc_id = str(svc[0].get("id") or svc[0].get("@id") or "")
            elif isinstance(svc, dict):
                svc_id = str(svc.get("id") or svc.get("@id") or "")
            res_id = str(body.get("id") or body.get("@id") or "")
            out.append(
                {
                    "index": idx,
                    "label": " ".join(_flatten_strings(c.get("label"))),
                    "width": c.get("width") or 0,
                    "height": c.get("height") or 0,
                    "service_id": svc_id,
                    "resource_id": res_id,
                    "thumbnail_url": "",
                }
            )
    return out


def _fetch_pil_image(session: requests.Session, url: str, timeout: float = 25.0) -> Optional[Image.Image]:
    if not is_allowed_url(url):
        return None
    r = http_get(session, url, headers={"User-Agent": USER_AGENT}, timeout=timeout, retries=2)
    if r is None:
        return None
    try:
        ct = r.headers.get("content-type", "").lower()
        if "image" not in ct and not r.content[:4].startswith((b"\xff\xd8\xff", b"\x89PNG")):
            return None
        im = Image.open(io.BytesIO(r.content))
        im.load()
        return im.convert("RGB")
    except Exception:
        return None
    finally:
        r.close()


def _candidate_preview_urls(canvas: Dict[str, Any]) -> List[str]:
    urls: List[str] = []
    thumb = canvas.get("thumbnail_url", "")
    svc = canvas.get("service_id", "").rstrip("/")
    res = canvas.get("resource_id", "")
    if svc and is_allowed_url(svc):
        # SBB Berlin returns 429 on non-full IIIF size transforms; use resource_id directly for SBB
        if "staatsbibliothek-berlin.de" in svc.lower():
            if res and is_allowed_url(res):
                urls.append(res)
            else:
                urls.append(f"{svc}/full/full/0/default.jpg")
        else:
            # Prefer >=500px preview so CLIP's min-250px gate never rejects tiny 150px thumbnails
            urls.append(f"{svc}/full/500,/0/default.jpg")
            urls.append(f"{svc}/full/!500,500/0/default.jpg")
    if thumb and is_allowed_url(thumb) and ",150/" not in thumb and "/150," not in thumb:
        urls.append(thumb)
    if res and is_allowed_url(res) and res not in urls:
        urls.append(res)
    return urls


def _candidate_hires_urls(canvas: Dict[str, Any], max_dim: int = 1800) -> List[str]:
    urls: List[str] = []
    svc = canvas.get("service_id", "").rstrip("/")
    res = canvas.get("resource_id", "")
    if svc and is_allowed_url(svc):
        low = svc.lower()
        if "slub-dresden.de" in low and ".tif.medium.jpg" in svc:
            large_svc = svc.replace(".tif.medium.jpg", ".tif.large.jpg")
            urls.append(f"{large_svc}/full/!{max_dim},{max_dim}/0/default.jpg")
            urls.append(f"{svc}/full/!{max_dim},{max_dim}/0/default.jpg")
        elif "staatsbibliothek-berlin.de" in low:
            if res and is_allowed_url(res):
                urls.append(res)
            urls.append(f"{svc}/full/full/0/default.jpg")
        else:
            urls.append(f"{svc}/full/!{max_dim},{max_dim}/0/default.jpg")
            urls.append(f"{svc}/full/{max_dim},/0/default.jpg")
            urls.append(f"{svc}/full/full/0/default.jpg")
    if res and is_allowed_url(res) and res not in urls:
        urls.append(res)
    return urls


def find_first_music_canvas(
    session: requests.Session,
    canvases: List[Dict[str, Any]],
    max_scan: int = 8,
    max_dim: int = 1800,
    clip_threshold: float = CLIP_MUSIC_THRESHOLD,
) -> Tuple[Optional[Image.Image], Optional[Dict[str, Any]], str, Dict[str, Any]]:
    """Find the first music canvas using CPU CLIP and return `(hires_img, canvas, img_url, clip_stats)`."""
    if not canvases:
        return None, None, "", {}

    # Scan canvases 1..max_scan-1 first (covers/bindings are almost always 0), then 0 as fallback
    n = min(len(canvases), max_scan)
    order = list(range(1, n)) + ([0] if n >= 1 else [])

    for idx in order:
        c = canvases[idx]
        preview_img: Optional[Image.Image] = None
        preview_url = ""
        for purl in _candidate_preview_urls(c):
            preview_img = _fetch_pil_image(session, purl, timeout=20.0)
            if preview_img is not None:
                preview_url = purl
                break
        if preview_img is None:
            continue

        ok, stats = is_music_page(preview_img, threshold=clip_threshold)
        if not ok:
            continue

        # Already high-res enough (e.g. SBB full/full)?
        if max(preview_img.size) >= 1200:
            return preview_img, c, preview_url, stats

        # Fetch high-res version of this confirmed music canvas
        for hurl in _candidate_hires_urls(c, max_dim=max_dim):
            hires_img = _fetch_pil_image(session, hurl, timeout=30.0)
            if hires_img is not None:
                return hires_img, c, hurl, stats

        # Fallback to preview if >= 800px
        if max(preview_img.size) >= 800:
            return preview_img, c, preview_url, stats

    return None, None, "", {}


class RismHarvestState:
    """Thread-safe resumable harvest state with atomic file writes & JSONL append."""

    def __init__(self, out_dir: Path, limit: int, resume: bool = True):
        self.out_dir = out_dir
        self.pages_dir = out_dir / "pages"
        self.manifest_path = out_dir / "manifest.jsonl"
        self.processed_path = out_dir / "processed_sources.txt"
        self.summary_path = out_dir / "summary.json"
        self.limit = limit
        self.lock = threading.Lock()
        self.start_time = time.time()

        self.pages_dir.mkdir(parents=True, exist_ok=True)
        self.seen_hashes: Set[str] = set()
        self.processed_sources: Set[str] = set()
        self.count = 0
        self.initial_count = 0
        self.by_institution: Dict[str, int] = {}
        self.portrait_count = 0
        self.landscape_count = 0

        # Diagnostic counters
        self.sources_inspected = 0
        self.pae_attempted = 0
        self.pae_succeeded = 0
        self.license_verified = 0
        self.license_rejected = 0
        self.no_iiif_url = 0
        self.no_music_canvas = 0
        self.eval_excluded = 0
        self.dup_skipped = 0
        self.error_reasons: Dict[str, int] = {}

        if resume and self.manifest_path.is_file():
            with self.manifest_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    p = Path(rec.get("image_path", ""))
                    if not p.is_file():
                        continue
                    sha = rec.get("image_sha256", "")
                    if sha:
                        self.seen_hashes.add(sha)
                    rid = rec.get("rism_id", "")
                    if rid:
                        self.processed_sources.add(rid)
                    self.count += 1
                    sig = rec.get("siglum", "unknown")
                    self.by_institution[sig] = self.by_institution.get(sig, 0) + 1
                    if rec.get("height", 0) >= rec.get("width", 0):
                        self.portrait_count += 1
                    else:
                        self.landscape_count += 1
        elif not resume:
            for p in (self.manifest_path, self.processed_path):
                if p.exists():
                    p.unlink()

        if resume and self.processed_path.is_file():
            with self.processed_path.open("r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s:
                        self.processed_sources.add(s)
        self.initial_count = self.count

    def is_done(self) -> bool:
        with self.lock:
            return self.count >= self.limit

    def mark_processed(self, rism_id: str) -> None:
        with self.lock:
            if rism_id in self.processed_sources:
                return
            self.processed_sources.add(rism_id)
            with self.processed_path.open("a", encoding="utf-8") as f:
                f.write(rism_id + "\n")
                f.flush()

    def bump(self, field: str, reason: str = "") -> None:
        with self.lock:
            setattr(self, field, getattr(self, field) + 1)
            if reason:
                key = reason.split(":")[0]
                self.error_reasons[key] = self.error_reasons.get(key, 0) + 1

    def record_pair(
        self,
        rism_id: str,
        siglum: str,
        canvas_idx: int,
        jpg_bytes: bytes,
        width: int,
        height: int,
        record_meta: Dict[str, Any],
    ) -> bool:
        sha = hashlib.sha256(jpg_bytes).hexdigest()
        with self.lock:
            if self.count >= self.limit:
                return False
            if sha in self.seen_hashes:
                self.dup_skipped += 1
                return False

            safe_sig = re.sub(r"[^A-Za-z0-9_-]+", "_", siglum)
            fname = f"{safe_sig}_{rism_id}_c{canvas_idx:02d}.jpg"
            final_path = self.pages_dir / fname
            tmp_path = self.pages_dir / (fname + ".tmp")
            tmp_path.write_bytes(jpg_bytes)
            os.replace(tmp_path, final_path)

            rec = dict(record_meta)
            rec.update(
                {
                    "rism_id": rism_id,
                    "siglum": siglum,
                    "canvas_index": canvas_idx,
                    "image_path": str(final_path),
                    "width": width,
                    "height": height,
                    "orientation": "portrait" if height >= width else "landscape",
                    "image_sha256": sha,
                    "rism_license": "CC-BY 4.0",
                }
            )

            with self.manifest_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()

            self.seen_hashes.add(sha)
            self.count += 1
            self.by_institution[siglum] = self.by_institution.get(siglum, 0) + 1
            if height >= width:
                self.portrait_count += 1
            else:
                self.landscape_count += 1
            return True

    def write_summary(self) -> Dict[str, Any]:
        with self.lock:
            elapsed = max(0.001, time.time() - self.start_time)
            new_pairs = max(0, self.count - self.initial_count)
            rate_per_min = (new_pairs / elapsed) * 60.0
            pae_rate = (self.pae_succeeded / self.pae_attempted) if self.pae_attempted else 0.0
            eta_5k_min = (5000.0 / rate_per_min) if rate_per_min > 0 else None
            summary = {
                "total_verified_pairs": self.count,
                "session_new_pairs": new_pairs,
                "limit": self.limit,
                "by_institution": dict(sorted(self.by_institution.items())),
                "portrait_count": self.portrait_count,
                "landscape_count": self.landscape_count,
                "sources_inspected": self.sources_inspected,
                "pae_attempted": self.pae_attempted,
                "pae_succeeded": self.pae_succeeded,
                "pae_conversion_rate": round(pae_rate, 4),
                "license_verified": self.license_verified,
                "license_rejected": self.license_rejected,
                "no_iiif_url": self.no_iiif_url,
                "no_music_canvas": self.no_music_canvas,
                "eval_excluded": self.eval_excluded,
                "dup_skipped": self.dup_skipped,
                "error_reasons": dict(sorted(self.error_reasons.items(), key=lambda kv: -kv[1])[:20]),
                "elapsed_sec": round(elapsed, 2),
                "pairs_per_min": round(rate_per_min, 2),
                "est_min_for_5000_pairs": round(eta_5k_min, 1) if eta_5k_min is not None else None,
            }
            tmp = self.summary_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self.summary_path)
            return summary


def process_source(
    source_url: str,
    siglum: str,
    state: RismHarvestState,
    assembly_bin: Path,
    eval_slugs: Set[str],
    max_dim: int,
    clip_threshold: float,
) -> Optional[str]:
    """Process one RISM manuscript source end-to-end."""
    if state.is_done():
        return None
    rism_id = source_url.rstrip("/").split("/")[-1]
    if rism_id in state.processed_sources:
        return None

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    state.bump("sources_inspected")

    try:
        r_src = http_get(session, source_url, headers=RISM_HEADERS, timeout=20.0, retries=2)
        if r_src is None:
            state.bump("no_iiif_url", "rism_fetch_failed")
            return None
        src_json = r_src.json()
        r_src.close()

        # Verify manuscript sourceType
        stype = src_json.get("sourceTypes", {}).get("sourceType", {}).get("type", "")
        if stype and "Manuscript" not in stype:
            return None

        # Eval-leakage protection
        label_str = " ".join(_flatten_strings(src_json.get("label")))
        creator_str = " ".join(_flatten_strings(src_json.get("creator")))
        excl, excl_reason = is_eval_excluded(f"{creator_str} {label_str}", rism_id, extra_slugs=eval_slugs)
        if excl:
            state.bump("eval_excluded", excl_reason)
            return None

        # 1. Extract allowlisted IIIF manifest URL
        manifest_url = extract_iiif_manifest_url(session, src_json, siglum)
        if not manifest_url:
            state.bump("no_iiif_url", "no_allowed_iiif")
            return None

        # 2. Fetch IIIF manifest & strictly verify PDM 1.0 / CC0 / PD license
        r_man = http_get(session, manifest_url, headers={"User-Agent": USER_AGENT}, timeout=20.0, retries=2)
        if r_man is None:
            state.bump("no_iiif_url", "manifest_fetch_failed")
            return None
        try:
            manifest = r_man.json()
        except Exception:
            state.bump("no_iiif_url", "manifest_bad_json")
            return None
        finally:
            r_man.close()

        lic_ok, lic_evidence = verify_iiif_license(manifest, manifest_url)
        if not lic_ok:
            state.bump("license_rejected", lic_evidence)
            return None
        state.bump("license_verified")

        # 3. Convert first valid PAE incipit -> MEI -> MusicXML -> bekern + bekern-decode
        incipits = src_json.get("incipits", {}).get("items", [])
        if not isinstance(incipits, list) or not incipits:
            state.bump("pae_attempted", "no_incipits")
            return None

        bekern_txt: Optional[str] = None
        inc_meta: Dict[str, Any] = {}
        last_err = ""
        state.bump("pae_attempted")
        for inc in incipits[:3]:
            bk, imeta, err = convert_incipit_to_bekern(session, inc, assembly_bin)
            if bk is not None:
                bekern_txt = bk
                inc_meta = imeta
                break
            last_err = err

        if bekern_txt is None:
            state.bump("error_reasons" if False else "sources_inspected")  # keep count clean below
            with state.lock:
                state.sources_inspected -= 1
                key = (last_err or "pae_failed").split(":")[0]
                state.error_reasons[key] = state.error_reasons.get(key, 0) + 1
            return None
        state.bump("pae_succeeded")

        # 4. Find first music canvas with CPU CLIP & download high-res image
        canvases = extract_canvases(manifest)
        hires_img, canvas, img_url, clip_stats = find_first_music_canvas(
            session, canvases, max_scan=8, max_dim=max_dim, clip_threshold=clip_threshold
        )
        if hires_img is None or canvas is None:
            state.bump("no_music_canvas", "no_music_canvas")
            return None

        jpg_bytes, w, h = resize_and_encode_jpeg(hires_img, max_dim=max_dim, quality=90)
        record_meta = {
            "source_url": source_url,
            "title": label_str[:300],
            "creator": creator_str[:200],
            "iiif_manifest_url": manifest_url,
            "image_url": img_url,
            "canvas_label": canvas.get("label", ""),
            "total_canvases": len(canvases),
            "license_verified": True,
            "license_evidence": lic_evidence,
            "clip_stats": clip_stats,
            "bekern": bekern_txt,
            **inc_meta,
        }
        saved = state.record_pair(rism_id, siglum, canvas["index"], jpg_bytes, w, h, record_meta)
        return rism_id if saved else None
    finally:
        state.mark_processed(rism_id)
        session.close()


def fetch_institution_source_urls(
    siglum: str,
    max_pages: int = 10,
    rows: int = 100,
) -> List[str]:
    """Deterministically fetch RISM manuscript source URLs for an institution."""
    fq_dig = INSTITUTION_CONFIGS.get(siglum, "has-iiif:true")
    session = requests.Session()
    urls: List[str] = []
    try:
        for page in range(1, max_pages + 1):
            params = [
                ("mode", "sources"),
                ("fq", f"holding-institution:{siglum}"),
                ("fq", fq_dig),
                ("fq", "has-incipits:true"),
                ("fq", "source-type:manuscript"),
                ("fq", "hide-source-contents:true"),
                ("fq", "hide-composite-volumes:true"),
                ("rows", str(rows)),
                ("page", str(page)),
            ]
            r = http_get(session, f"{RISM_API_BASE}/search", params=params, headers=RISM_HEADERS, timeout=25.0)
            if r is None:
                break
            data = r.json()
            r.close()
            items = data.get("items") or []
            if not items:
                break
            for it in items:
                u = it.get("id")
                if u:
                    urls.append(str(u))
            total_pages = int(data.get("view", {}).get("totalPages") or 1)
            if page >= total_pages:
                break
    finally:
        session.close()
    return urls


def main() -> int:
    ap = argparse.ArgumentParser(description="Harvest RISM CC-BY 4.0 x PDM 1.0 IIIF manuscripts (CPU-only).")
    ap.add_argument("--limit", type=int, default=200, help="Target number of verified pairs to harvest.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help="Output directory.")
    ap.add_argument(
        "--institutions",
        type=str,
        default="D-Dl,D-B,D-Mbs",
        help="Comma-separated RISM sigla (e.g. D-Dl,D-B,D-Mbs,PL-Wn,US-Wc,A-Wn).",
    )
    ap.add_argument("--per-inst-cap", type=int, default=0, help="Optional max pairs per institution (0=balanced).")
    ap.add_argument("--workers", type=int, default=6, help="Worker threads (CPU + network only).")
    ap.add_argument("--max-dim", type=int, default=1800, help="Max image dimension (px) for saved JPEGs.")
    ap.add_argument("--clip-threshold", type=float, default=CLIP_MUSIC_THRESHOLD, help="CLIP music threshold.")
    ap.add_argument("--max-search-pages", type=int, default=6, help="Max RISM search pages (100/page) per institution.")
    ap.add_argument("--assembly-bin", type=Path, default=DEFAULT_ASSEMBLY_BIN, help="Path to assembly-serialize.exe.")
    ap.add_argument("--resume", action="store_true", help="Resume from existing manifest and processed list.")
    args = ap.parse_args()

    if not args.assembly_bin.is_file():
        print(f"ERROR: assembly-serialize.exe not found at {args.assembly_bin}", file=sys.stderr)
        return 1

    sigla = [s.strip() for s in args.institutions.split(",") if s.strip()]
    for s in sigla:
        if s not in INSTITUTION_CONFIGS:
            print(f"ERROR: unsupported institution siglum {s!r}; allowed={list(INSTITUTION_CONFIGS)}", file=sys.stderr)
            return 1

    eval_slugs = load_eval_slugs()
    state = RismHarvestState(args.out, limit=args.limit, resume=args.resume)
    print(
        f"[init] out={args.out} limit={args.limit} resume={args.resume} "
        f"existing={state.count} processed={len(state.processed_sources)} sigla={sigla}",
        flush=True,
    )

    if state.is_done():
        summary = state.write_summary()
        print("[done] already reached limit:", json.dumps(summary, indent=2), flush=True)
        return 0

    # Pre-warm CPU CLIP model once on main thread so workers never race
    from harvest_unlabeled_10k import get_clip_classifier

    t0_clip = time.time()
    get_clip_classifier()
    print(f"[clip] CPU CLIP classifier ready in {time.time() - t0_clip:.1f}s", flush=True)

    # Interleave candidate URLs across institutions for balanced coverage
    per_inst_urls: Dict[str, List[str]] = {}
    for sig in sigla:
        urls = fetch_institution_source_urls(sig, max_pages=args.max_search_pages, rows=100)
        # Filter already-processed
        urls = [u for u in urls if u.rstrip("/").split("/")[-1] not in state.processed_sources]
        per_inst_urls[sig] = urls
        print(f"[search] {sig}: queued {len(urls)} unprocessed manuscript candidates", flush=True)

    # Round-robin interleave across institutions
    work_queue: List[Tuple[str, str]] = []
    max_len = max((len(v) for v in per_inst_urls.values()), default=0)
    for i in range(max_len):
        for sig in sigla:
            if i < len(per_inst_urls[sig]):
                work_queue.append((per_inst_urls[sig][i], sig))

    print(f"[harvest] starting with {len(work_queue)} interleaved candidates across {args.workers} CPU workers", flush=True)

    last_log = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        in_flight: Set[concurrent.futures.Future[Optional[str]]] = set()
        idx = 0
        max_in_flight = args.workers * 3

        while (idx < len(work_queue) or in_flight) and not state.is_done():
            while idx < len(work_queue) and len(in_flight) < max_in_flight and not state.is_done():
                src_url, sig = work_queue[idx]
                idx += 1
                if args.per_inst_cap > 0 and state.by_institution.get(sig, 0) >= args.per_inst_cap:
                    continue
                fut = ex.submit(
                    process_source,
                    src_url,
                    sig,
                    state,
                    args.assembly_bin,
                    eval_slugs,
                    args.max_dim,
                    args.clip_threshold,
                )
                in_flight.add(fut)

            if not in_flight:
                break

            done, in_flight = concurrent.futures.wait(
                in_flight, timeout=1.0, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for fut in done:
                try:
                    fut.result()
                except Exception as e:
                    state.bump("error_reasons" if False else "no_iiif_url", f"worker_exc:{type(e).__name__}")

            now = time.time()
            if now - last_log >= 15.0:
                s = state.write_summary()
                print(
                    f"[progress] {s['total_verified_pairs']}/{args.limit} pairs "
                    f"({s['pairs_per_min']:.1f}/min) | inst={s['by_institution']} | "
                    f"pae={s['pae_succeeded']}/{s['pae_attempted']} ({s['pae_conversion_rate']*100:.1f}%) | "
                    f"lic_ok={s['license_verified']} lic_rej={s['license_rejected']} "
                    f"no_music={s['no_music_canvas']}",
                    flush=True,
                )
                last_log = now

        # Cancel remaining queued futures cleanly
        for fut in in_flight:
            fut.cancel()

    summary = state.write_summary()
    print("[finished] summary:\n" + json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
