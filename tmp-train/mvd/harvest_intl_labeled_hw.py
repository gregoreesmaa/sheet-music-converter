#!/usr/bin/env python3
"""International Labeled Handwritten HW Harvester (CPU-only, Commercial-Safe).

Harvests strictly commercial-safe (PDM 1.0 / CC0 / CC-BY 4.0) labeled real
handwritten music manuscripts from international public archives outside
Germany/US/Poland/Austria (and excluding BnF Gallica / NC / anti-bot gated sites):

  - Switzerland (`e-manuscripta.ch`, verified per-item `PDM 1.0`):
      * `CH-Bu` (Universitätsbibliothek Basel — 1,572 RISM manuscripts w/ PAE incipits)
      * `CH-SO` (Zentralbibliothek Solothurn — e-manuscripta.ch holdings)
      * Direct `e-manuscripta.ch` SRU search for OpenScore Lieder CC0 full-work matches

Archive Probe Decisions (enforced by strict license & access-control gates):
  - `e-manuscripta.ch` (CH-Bu, CH-SO): PASS — native IIIF v2 + verifiable per-item
    `creativecommons.org/publicdomain/mark/1.0` URI on landing page.
  - `iiif.rism.digital` (CH-Zz, CH-E, CH-BEl, CH-Lz, CH-BEk, CH-SGv, B-Bc):
    REJECTED — manifests carry zero license/rights metadata.
  - `alvin-portal.org` (S-Uu, S-L): SKIPPED — gated by Anubis PoW anti-bot challenge.
  - `bdh-rd.bne.es` (E-Mn): SKIPPED — HTTP 403 WAF on automated requests.
  - `iiif.bodleian.ox.ac.uk` (GB-Ob): REJECTED — `CC BY-NC 4.0`.
  - `archive.org` (GB-WCr): REJECTED — `CC BY-NC-ND 4.0`.

Outputs:
  - Part A (OpenScore Lieder full-work matches, if any):
      Saved under `D:\\tmp\\rism-pdm-hw-intl\\openscore_matches\\` and appended to
      `D:\\tmp\\real-hw-corpus\\train_works.jsonl` (0 eval overlap, bekern-decode rc=0).
  - Part B (RISM-labeled opening-incipit manuscripts):
      Saved under `D:\\tmp\\rism-pdm-hw-intl\\`:
        * `pages/{siglum}_{rism_id}_c{idx:02d}.jpg` (<= 1800 px RGB JPEG)
        * `mei/{siglum}_{rism_id}.mei`
        * `musicxml/{siglum}_{rism_id}.musicxml`
        * `bekern/{siglum}_{rism_id}-bekern.txt` (post-I1 canonical, bekern-decode rc=0)
        * `manifest.jsonl`, `summary.json`, `USAGE.md`

Usage:
    .\\.venv\\Scripts\\python.exe tmp-train/mvd/harvest_intl_labeled_hw.py ^
        --limit 300 --workers 6 --resume
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
from urllib.parse import quote, urlparse

import music21
import requests
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import recanon_i1  # noqa: E402
from harvest_rism_pdm import (  # noqa: E402
    DEFAULT_ASSEMBLY_BIN,
    RISM_API_BASE,
    RISM_HEADERS,
    _flatten_strings,
    extract_canvases,
    http_get,
    patch_mei_for_music21,
)
from harvest_pdm_openscore import (  # noqa: E402
    DEFAULT_LIEDER_DIR,
    OpenScoreSong,
    clean_openscore_title,
    extract_catalog_tokens,
    extract_mxl_to_musicxml,
    is_held_out_hw,
    load_eval_titles,
    normalize_text,
    slugify,
    title_variants,
)
from harvest_unlabeled_10k import (  # noqa: E402
    CLIP_MUSIC_THRESHOLD,
    get_clip_classifier,
    is_eval_excluded,
    is_music_page,
    load_eval_slugs,
    resize_and_encode_jpeg,
)

USER_AGENT = "sheet-music-converter-research/1.0 (intl-PDM-1.0-harvester; CPU-only)"
DEFAULT_OUT_DIR = Path(r"D:\tmp\rism-pdm-hw-intl")
REAL_HW_CORPUS_ROOT = Path(r"D:\tmp\real-hw-corpus")
SUITE31_HW_MANIFEST = REPO_ROOT / "data-gen" / "score-corpus" / "suite31" / "suite31_hw_manifest.json"

# Strictly allowlisted international commercial-safe IIIF domains
INTL_ALLOWED_DOMAIN_SUFFIXES: Tuple[str, ...] = (
    "e-manuscripta.ch",
)

# Explicit hard blocklist (defense-in-depth)
BLOCKED_DOMAIN_SUBSTRINGS: Tuple[str, ...] = (
    "gallica.bnf.fr",
    "bnf.fr",
    "bodleian.ox.ac.uk",
    "alvin-portal.org",
)

# Strict License Gate regexes
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
    re.compile(r"creativecommons\.org/licenses/by-nc", re.I),
)

ACCEPT_LICENSE_URI_PATTERNS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?:https?:)?//creativecommons\.org/publicdomain/mark/1\.0[^\s\"'<>]*", re.I), "PDM 1.0"),
    (re.compile(r"(?:https?:)?//creativecommons\.org/publicdomain/zero/1\.0[^\s\"'<>]*", re.I), "CC0 1.0"),
    (re.compile(r"(?:https?:)?//creativecommons\.org/licenses/by/4\.0[^\s\"'<>]*", re.I), "CC-BY 4.0"),
    (re.compile(r"(?:https?:)?//rightsstatements\.org/vocab/pdm/[^\s\"'<>]*", re.I), "PDM 1.0"),
)

# International RISM institution metadata
INTL_INSTITUTIONS: Dict[str, Dict[str, str]] = {
    "CH-Bu": {
        "institution": "Universitätsbibliothek Basel",
        "country": "CH",
        "archive": "e-manuscripta.ch",
    },
    "CH-SO": {
        "institution": "Zentralbibliothek Solothurn",
        "country": "CH",
        "archive": "e-manuscripta.ch",
    },
}


def is_intl_allowed_url(url: str) -> bool:
    if not url or not url.startswith(("http://", "https://")):
        return False
    low = url.lower()
    for blk in BLOCKED_DOMAIN_SUBSTRINGS:
        if blk in low:
            return False
    host = (urlparse(low).hostname or "").strip(".")
    return any(host == suf or host.endswith("." + suf) for suf in INTL_ALLOWED_DOMAIN_SUFFIXES)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def normalize_uri(raw: str) -> str:
    u = raw.strip().rstrip(".,;)\"'<>")
    if u.startswith("//"):
        return "https:" + u
    return u


def verify_emanuscripta_license(
    session: requests.Session,
    manifest: Dict[str, Any],
    manifest_url: str,
    doi_url: str = "",
    prefetched_html: str = "",
    prefetched_url: str = "",
) -> Tuple[bool, str, str, str]:
    """Verify per-item PDM 1.0 / CC0 / CC-BY 4.0 license for an e-manuscripta.ch item.

    Returns `(ok, license_normalized, license_uri, landing_page_url)`.
    """
    if not is_intl_allowed_url(manifest_url):
        return False, "disallowed_manifest_host", "", ""

    # 1. Check manifest fields for any forbidden NC/InC/Gallica tokens
    manifest_chunks: List[str] = []
    for k in ("license", "rights", "attribution", "requiredStatement"):
        if k in manifest:
            manifest_chunks.extend(_flatten_strings(manifest[k]))
    manifest_doi = doi_url
    for md in manifest.get("metadata") or []:
        if not isinstance(md, dict):
            continue
        lbl = " ".join(_flatten_strings(md.get("label"))).lower()
        val_s = " ".join(_flatten_strings(md.get("value")))
        if any(k in lbl for k in ("license", "lizenz", "rights", "rechte", "copyright", "nutzung")):
            manifest_chunks.append(val_s)
        if not manifest_doi and "doi.org/10.7891/e-manuscripta-" in val_s:
            m_doi = re.search(r"https?://doi\.org/10\.7891/e-manuscripta-\d+", val_s)
            if m_doi:
                manifest_doi = m_doi.group(0)

    joined_manifest = " | ".join(manifest_chunks)
    for pat in REJECT_LICENSE_PATTERNS:
        m = pat.search(joined_manifest)
        if m:
            return False, f"rejected_manifest:{m.group(0)}", "", ""

    # If manifest itself already carries an explicit accepted license URI, accept immediately
    for pat, norm in ACCEPT_LICENSE_URI_PATTERNS:
        m = pat.search(joined_manifest)
        if m:
            return True, norm, normalize_uri(m.group(0)), manifest_doi or manifest_url

    def _check_html(html: str, final_url: str) -> Optional[Tuple[bool, str, str, str]]:
        if not is_intl_allowed_url(final_url):
            return None
        cc_links = re.findall(
            r"(?:https?:)?//(?:creativecommons\.org|rightsstatements\.org)/[^\s\"'<>]+",
            html,
            re.I,
        )
        cc_joined = " | ".join(cc_links)
        for pat in REJECT_LICENSE_PATTERNS:
            m = pat.search(cc_joined)
            if m:
                return False, f"rejected_page:{m.group(0)}", "", final_url
        for pat, norm in ACCEPT_LICENSE_URI_PATTERNS:
            m = pat.search(cc_joined)
            if m:
                return True, norm, normalize_uri(m.group(0)), final_url
        return None

    # 2a. Reuse prefetched DOI landing-page HTML if already fetched (avoids duplicate HTTP round-trip)
    if prefetched_html and prefetched_url:
        res_pf = _check_html(prefetched_html, prefetched_url)
        if res_pf is not None:
            return res_pf

    # 2b. Fetch the e-manuscripta.ch landing page (via DOI redirect or content/titleinfo)
    candidate_pages: List[str] = []
    if manifest_doi:
        candidate_pages.append(manifest_doi)
    m_id = re.search(r"/i3f/v20/(\d+)/manifest", manifest_url)
    if m_id:
        candidate_pages.append(f"https://www.e-manuscripta.ch/content/titleinfo/{m_id.group(1)}")

    for page_url in candidate_pages:
        r = http_get(session, page_url, headers={"User-Agent": USER_AGENT}, timeout=20.0, retries=2, allow_redirects=True)
        if r is None:
            continue
        html = r.text
        final_url = r.url
        r.close()
        res_pg = _check_html(html, final_url)
        if res_pg is not None:
            return res_pg

    return False, "no_verified_license_uri", "", manifest_doi


def extract_emanuscripta_manifest(
    session: requests.Session,
    src_text: str,
) -> Tuple[Optional[str], str, str, str]:
    """Extract an e-manuscripta.ch IIIF manifest URL + optional DOI URL + prefetched landing HTML."""
    urls = sorted(set(re.findall(r'https?://[^\s"<>\\]+', src_text)))
    doi_url = ""
    for u in urls:
        if "doi.org/10.7891/e-manuscripta-" in u.lower():
            doi_url = u
            break

    # 1. Direct e-manuscripta IIIF manifest URL
    for u in urls:
        if "e-manuscripta.ch/i3f/" in u.lower() and "manifest" in u.lower() and is_intl_allowed_url(u):
            return u, doi_url, "", ""

    # 2. Direct e-manuscripta content/titleinfo/{id} -> i3f/v20/{id}/manifest
    for u in urls:
        if "e-manuscripta.ch" in u.lower():
            m = re.search(r"/content/(?:titleinfo|pageview)/(\d+)", u)
            if m:
                cand = f"https://www.e-manuscripta.ch/i3f/v20/{m.group(1)}/manifest"
                return cand, doi_url, "", ""

    # 3. Resolve DOI -> e-manuscripta.ch landing page -> extract IIIF manifest link
    if doi_url:
        r = http_get(session, doi_url, headers={"User-Agent": USER_AGENT}, timeout=20.0, retries=2, allow_redirects=True)
        if r is not None:
            html = r.text
            final_url = r.url
            r.close()
            m_iiif = re.search(r"https?://www\.e-manuscripta\.ch/i3f/v20/\d+/manifest", html)
            if m_iiif and is_intl_allowed_url(m_iiif.group(0)):
                return m_iiif.group(0), doi_url, html, final_url

    return None, doi_url, "", ""


def convert_incipit_full(
    session: requests.Session,
    incipit_item: Dict[str, Any],
    assembly_bin: Path,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Fetch RISM MEI -> patch -> music21 -> MusicXML -> canonical post-I1 bekern + bekern-decode rc=0.

    Returns `(artifact_dict_or_None, error_reason)`.
    """
    inc_id = str(incipit_item.get("id", "")).strip()
    props = incipit_item.get("properties") or {}
    if not inc_id or not props.get("notation"):
        return None, "missing_pae_notation"

    mei_url = inc_id.rstrip("/") + "/mei"
    for enc in incipit_item.get("encodings") or []:
        if enc.get("format") == "application/mei+xml" and enc.get("url"):
            mei_url = str(enc["url"])
            break

    r_mei = http_get(session, mei_url, headers={"User-Agent": USER_AGENT}, timeout=20.0, retries=2)
    if r_mei is None:
        return None, "mei_fetch_failed"
    mei_raw = r_mei.text
    r_mei.close()

    try:
        mei_patched = patch_mei_for_music21(mei_raw, props)
    except ET.ParseError as e:
        return None, f"mei_xml_parse_err:{e}"

    try:
        score = music21.converter.parseData(mei_patched, format="mei")
    except Exception as e:
        return None, f"music21_mei_err:{type(e).__name__}:{str(e)[:100]}"

    with tempfile.TemporaryDirectory(prefix="intl_bk_") as td:
        td_path = Path(td)
        mxml_path = td_path / "incipit.musicxml"
        try:
            score.write("musicxml", fp=str(mxml_path))
        except Exception as e:
            return None, f"musicxml_write_err:{type(e).__name__}:{str(e)[:100]}"

        musicxml_str = mxml_path.read_text(encoding="utf-8")
        # Count bars from first part in MusicXML
        try:
            mx_root = ET.fromstring(musicxml_str)
            parts = mx_root.findall("part")
            n_bars = len(parts[0].findall("measure")) if parts else 0
        except Exception:
            n_bars = 0

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
            return None, f"bekern_rc{r1.returncode}:{r1.stderr.strip()[:120]}"

        bk_path = td_path / "incipit-bekern.txt"
        if not bk_path.is_file():
            return None, "bekern_missing_output"
        bekern_txt = bk_path.read_text(encoding="utf-8").strip()
        toks = bekern_txt.split()
        if len(toks) < 6:
            return None, f"bekern_too_short:{len(toks)}"

        # Enforce post-I1 canonical form
        if not recanon_i1.is_stream_canonical_i1(bekern_txt):
            new_toks, changed = recanon_i1.canonicalize_bekern_tokens(toks)
            if changed:
                bekern_txt = " ".join(new_toks)
                toks = new_toks
                bk_path.write_text(bekern_txt + "\n", encoding="utf-8")
            if not recanon_i1.is_stream_canonical_i1(bekern_txt):
                return None, "non_canonical_i1"

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
        if r2.returncode != 0 or not dec_path.is_file():
            return None, f"bekern_decode_rc{r2.returncode}:{r2.stderr.strip()[:120]}"

    if n_bars == 0:
        n_bars = sum(1 for t in toks if t.startswith("="))

    return (
        {
            "mei_xml": mei_patched,
            "musicxml": musicxml_str,
            "bekern": bekern_txt,
            "n_bars": n_bars,
            "n_tokens": len(toks),
            "bekern_decode_rc": 0,
            "incipit_id": inc_id,
            "mei_url": mei_url,
            "pae_clef": props.get("clef", ""),
            "pae_keysig": props.get("keysig", ""),
            "pae_timesig": props.get("timesig", ""),
            "pae_notation": props.get("notation", ""),
        },
        "",
    )


def _fetch_intl_image(session: requests.Session, url: str, timeout: float = 25.0) -> Optional[Image.Image]:
    if not is_intl_allowed_url(url):
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


def find_first_intl_music_canvas(
    session: requests.Session,
    canvases: List[Dict[str, Any]],
    max_scan: int = 8,
    max_dim: int = 1800,
    clip_threshold: float = CLIP_MUSIC_THRESHOLD,
) -> Tuple[Optional[Image.Image], Optional[Dict[str, Any]], str, Dict[str, Any]]:
    """Find first music canvas on an international IIIF manifest via CPU CLIP."""
    if not canvases:
        return None, None, "", {}

    n = min(len(canvases), max_scan)
    order = list(range(1, n)) + ([0] if n >= 1 else [])

    for idx in order:
        c = canvases[idx]
        svc = c.get("service_id", "").rstrip("/")
        res = c.get("resource_id", "")
        preview_urls: List[str] = []
        if svc and is_intl_allowed_url(svc):
            preview_urls.append(f"{svc}/full/500,/0/default.jpg")
            preview_urls.append(f"{svc}/full/!500,500/0/default.jpg")
        if res and is_intl_allowed_url(res) and res not in preview_urls:
            preview_urls.append(res)

        preview_img: Optional[Image.Image] = None
        preview_url = ""
        for purl in preview_urls:
            preview_img = _fetch_intl_image(session, purl, timeout=20.0)
            if preview_img is not None:
                preview_url = purl
                break
        if preview_img is None:
            continue

        ok, stats = is_music_page(preview_img, threshold=clip_threshold)
        if not ok:
            continue

        if max(preview_img.size) >= 1200:
            return preview_img, c, preview_url, stats

        hires_urls: List[str] = []
        if svc and is_intl_allowed_url(svc):
            hires_urls.append(f"{svc}/full/!{max_dim},{max_dim}/0/default.jpg")
            hires_urls.append(f"{svc}/full/{max_dim},/0/default.jpg")
            hires_urls.append(f"{svc}/full/full/0/default.jpg")
        if res and is_intl_allowed_url(res) and res not in hires_urls:
            hires_urls.append(res)

        for hurl in hires_urls:
            hires_img = _fetch_intl_image(session, hurl, timeout=30.0)
            if hires_img is not None:
                return hires_img, c, hurl, stats

        if max(preview_img.size) >= 800:
            return preview_img, c, preview_url, stats

    return None, None, "", {}


# ---------------------------------------------------------------------------
# Part A: OpenScore Lieder CC0 full-work matching on e-manuscripta.ch
# ---------------------------------------------------------------------------

def index_all_openscore_lieder(lieder_dir: Path) -> Dict[str, List[OpenScoreSong]]:
    """Index ALL 1,462 CC0 OpenScore Lieder works grouped by composer folder."""
    eval_titles = load_eval_titles()
    by_composer: Dict[str, List[OpenScoreSong]] = {}
    if not lieder_dir.is_dir():
        return by_composer

    for cdir in sorted(p for p in lieder_dir.iterdir() if p.is_dir()):
        folder = cdir.name
        parts_name = folder.split(",_")
        surname = normalize_text(parts_name[0]) if parts_name else normalize_text(folder)
        given = normalize_text(parts_name[1].split("_")[0]) if len(parts_name) > 1 else ""
        composer_display = folder.replace(",_", ", ").replace("_", " ")
        songs: List[OpenScoreSong] = []
        for mxl in sorted(cdir.rglob("*.mxl")):
            rel = mxl.relative_to(cdir)
            parts = rel.parts
            collection = parts[0].replace("_", " ") if len(parts) >= 3 else ""
            raw_title = parts[-2] if len(parts) >= 2 else mxl.stem
            clean_t = clean_openscore_title(raw_title)
            norm_t = normalize_text(clean_t)
            if len(norm_t) < 4:
                continue
            cats = extract_catalog_tokens(f"{collection} {raw_title}")
            ov = any(norm_t in et or et in norm_t for et in eval_titles)
            songs.append(
                OpenScoreSong(
                    composer_folder=folder,
                    composer_display=composer_display,
                    surname=surname,
                    given=given,
                    collection=collection,
                    raw_title=raw_title,
                    clean_title=clean_t,
                    norm_title=norm_t,
                    mxl_path=mxl,
                    catalog_tokens=cats,
                    eval_overlap=ov,
                )
            )
        if songs:
            by_composer[folder] = songs
    return by_composer


def search_emanuscripta_sru(session: requests.Session, query: str, max_records: int = 50) -> List[Dict[str, Any]]:
    """Query e-manuscripta.ch SRU 1.2 (Dublin Core) for music manuscripts."""
    url = (
        "https://www.e-manuscripta.ch/sru?version=1.2&operation=searchRetrieve"
        f"&query={quote(query)}&maximumRecords={max_records}&recordSchema=dc"
    )
    r = http_get(session, url, headers={"User-Agent": USER_AGENT}, timeout=20.0, retries=2)
    if r is None:
        return []
    xml_txt = r.text
    r.close()
    try:
        root = ET.fromstring(xml_txt)
    except ET.ParseError:
        return []

    dc_ns = {"dc": "http://purl.org/dc/elements/1.1/"}
    records: List[Dict[str, Any]] = []
    for rec in root.findall(".//{http://www.loc.gov/zing/srw/}recordData"):
        titles = [el.text.strip() for el in rec.findall(".//dc:title", dc_ns) if el.text and el.text.strip()]
        creators = [el.text.strip() for el in rec.findall(".//dc:creator", dc_ns) if el.text and el.text.strip()]
        descs = [el.text.strip() for el in rec.findall(".//dc:description", dc_ns) if el.text and el.text.strip()]
        idents = [el.text.strip() for el in rec.findall(".//dc:identifier", dc_ns) if el.text and el.text.strip()]
        title_all = " ".join(titles)
        desc_all = " ".join(descs)
        # Skip letters / correspondence / postcards (not music scores)
        low = f"{title_all} {desc_all}".lower()
        if any(w in low for w in ("brief an", "eigenhändiger brief", "postkarte", "korrespondenz", "autograph-brief", "widmungsbrief")):
            continue
        em_id = ""
        doi_url = ""
        for ident in idents:
            m_doi = re.search(r"https?://doi\.org/10\.7891/e-manuscripta-\d+", ident)
            if m_doi:
                doi_url = m_doi.group(0)
            m_id = re.search(r"/content/titleinfo/(\d+)", ident)
            if m_id:
                em_id = m_id.group(1)
        if not em_id and not doi_url:
            continue
        records.append(
            {
                "title": title_all,
                "creator": " ; ".join(creators),
                "description": desc_all,
                "em_id": em_id,
                "doi_url": doi_url,
            }
        )
    return records


def run_part_a_openscore(
    out_dir: Path,
    lieder_dir: Path,
    assembly_bin: Path,
    eval_slugs: Set[str],
) -> List[Dict[str, Any]]:
    """Probe e-manuscripta.ch for any PDM 1.0 full-work OpenScore Lieder matches."""
    print("[Part A] Indexing all CC0 OpenScore Lieder works ...", flush=True)
    by_composer = index_all_openscore_lieder(lieder_dir)
    total_songs = sum(len(v) for v in by_composer.values())
    print(f"[Part A] Indexed {total_songs} CC0 Lieder across {len(by_composer)} composers.", flush=True)

    # Load existing train_works.jsonl + suite31_hw_manifest.json to prevent duplicates & eval leakage
    existing_train_ids: Set[str] = set()
    existing_train_titles: Set[Tuple[str, str]] = set()
    train_jsonl = REAL_HW_CORPUS_ROOT / "train_works.jsonl"
    if train_jsonl.is_file():
        for line in train_jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                existing_train_ids.add(rec.get("work_id", ""))
                existing_train_titles.add((normalize_text(rec.get("composer", "")), normalize_text(rec.get("title", ""))))
            except Exception:
                pass

    eval_ids: Set[str] = set()
    if SUITE31_HW_MANIFEST.is_file():
        ev_data = json.loads(SUITE31_HW_MANIFEST.read_text(encoding="utf-8"))
        for p in ev_data.get("pieces", []):
            for k in ("slug", "work_id"):
                if p.get(k):
                    eval_ids.add(p[k])
                    if p[k].startswith("hw-"):
                        eval_ids.add(p[k][3:])

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    added_works: List[Dict[str, Any]] = []
    matches_root = out_dir / "openscore_matches"

    # Focus on composers with >= 3 OpenScore Lieder
    candidate_composers = [(f, s) for f, s in by_composer.items() if len(s) >= 3]
    print(f"[Part A] Querying e-manuscripta.ch SRU across {len(candidate_composers)} OpenScore composers ...", flush=True)

    for folder, songs in candidate_composers:
        surname = songs[0].surname
        given = songs[0].given
        if len(surname) < 3:
            continue
        recs = search_emanuscripta_sru(session, f'"{surname}"', max_records=50)
        for rec in recs:
            comp_norm = normalize_text(rec["creator"])
            if surname not in comp_norm:
                continue
            if given and len(given) >= 3 and given not in comp_norm:
                continue
            title_norm = normalize_text(rec["title"])
            if len(title_norm) < 4 or is_held_out_hw(comp_norm, title_norm):
                continue

            # Check for song title match
            matched_song: Optional[OpenScoreSong] = None
            for song in songs:
                if song.eval_overlap:
                    continue
                for v in title_variants(song):
                    if len(v) < 5:
                        continue
                    if v == title_norm or (len(v) >= 10 and " " in v and re.search(rf"\b{re.escape(v)}\b", title_norm)):
                        matched_song = song
                        break
                if matched_song:
                    break
            if matched_song is None:
                continue

            work_id = slugify(f"{matched_song.surname}-{matched_song.clean_title}-ch-eman")
            if work_id in existing_train_ids or work_id in eval_ids:
                continue
            excl, _ = is_eval_excluded(f"{matched_song.composer_display} {matched_song.clean_title}", work_id, extra_slugs=eval_slugs)
            if excl:
                continue

            # Resolve IIIF manifest + verify PDM 1.0
            manifest_url = f"https://www.e-manuscripta.ch/i3f/v20/{rec['em_id']}/manifest" if rec["em_id"] else ""
            pf_html, pf_url = "", ""
            if not manifest_url and rec["doi_url"]:
                manifest_url, _, pf_html, pf_url = extract_emanuscripta_manifest(session, rec["doi_url"])
            if not manifest_url:
                continue
            rm = http_get(session, manifest_url, headers={"User-Agent": USER_AGENT}, timeout=20.0, retries=2)
            if rm is None:
                continue
            try:
                manifest = rm.json()
            except Exception:
                rm.close()
                continue
            rm.close()

            lic_ok, lic_norm, lic_uri, landing_url = verify_emanuscripta_license(
                session, manifest, manifest_url, doi_url=rec["doi_url"], prefetched_html=pf_html, prefetched_url=pf_url
            )
            if not lic_ok:
                continue

            canvases = extract_canvases(manifest)
            if not (2 <= len(canvases) <= 24):
                continue

            # Download music canvases (CLIP verified)
            page_dir = matches_root / "pages" / work_id
            page_paths: List[str] = []
            page_dims: List[List[int]] = []
            n_port = 0
            n_land = 0
            for c in canvases:
                svc = c.get("service_id", "").rstrip("/")
                if not svc or not is_intl_allowed_url(svc):
                    continue
                img = _fetch_intl_image(session, f"{svc}/full/!1800,1800/0/default.jpg", timeout=25.0)
                if img is None:
                    continue
                ok_m, _ = is_music_page(img, threshold=CLIP_MUSIC_THRESHOLD)
                if not ok_m:
                    continue
                jpg_b, w, h = resize_and_encode_jpeg(img, max_dim=1800, quality=92)
                p_out = page_dir / f"page_{len(page_paths)+1:03d}.jpg"
                atomic_write_bytes(p_out, jpg_b)
                page_paths.append(str(p_out))
                page_dims.append([w, h])
                if h >= w:
                    n_port += 1
                else:
                    n_land += 1
            if len(page_paths) < 2:
                continue

            # Extract MusicXML and verify bekern + bekern-decode rc=0
            xml_out = matches_root / "xml" / f"{work_id}.musicxml"
            extract_mxl_to_musicxml(matched_song.mxl_path, xml_out)
            from incorporate_openscore_hw import convert_and_verify_bekern

            bk_ok, bk_toks, bk_status, bk_path = convert_and_verify_bekern(
                xml_out, REAL_HW_CORPUS_ROOT / "bekern", work_id, end_idx=None
            )
            if not bk_ok or bk_path is None:
                continue

            train_rec = {
                "work_id": work_id,
                "source": "openscore-lieder-pdm-intl",
                "composer": matched_song.composer_display,
                "title": matched_song.clean_title,
                "library": "e-manuscripta.ch (Switzerland)",
                "country": "CH",
                "license": f"{lic_norm} (e-manuscripta.ch, {lic_uri}) + CC0-1.0 (OpenScore Lieder)",
                "rights": lic_norm,
                "license_uri": lic_uri,
                "xml_license": "CC0-1.0 (OpenScore Lieder)",
                "material_type": "Autograph/Manuscript (e-manuscripta.ch)",
                "orientation": "portrait" if n_port >= n_land else "landscape",
                "n_pages": len(page_paths),
                "n_portrait_pages": n_port,
                "n_landscape_pages": n_land,
                "page_paths": page_paths,
                "page_dims": page_dims,
                "xml_path": str(xml_out),
                "bekern_path": str(bk_path),
                "bekern_ok": True,
                "bekern_tokens": bk_toks,
                "bekern_status": bk_status,
                "rism_url": landing_url or rec["doi_url"],
                "iiif_manifest": manifest_url,
            }
            with train_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(train_rec, ensure_ascii=False) + "\n")
                f.flush()
            existing_train_ids.add(work_id)
            added_works.append(train_rec)
            print(f"[Part A] ADDED full-work match: {work_id} ({len(page_paths)} pages, {bk_toks} bekern toks, {lic_uri})", flush=True)

    session.close()
    print(f"[Part A] Finished: {len(added_works)} verified OpenScore Lieder full-work matches found on e-manuscripta.ch.", flush=True)
    return added_works


# ---------------------------------------------------------------------------
# Part B: RISM-Labeled Opening-Incipit International Harvester
# ---------------------------------------------------------------------------

class IntlHarvestState:
    """Thread-safe resumable state with atomic writes for Part B."""

    def __init__(self, out_dir: Path, limit: int, resume: bool = True):
        self.out_dir = out_dir
        self.pages_dir = out_dir / "pages"
        self.mei_dir = out_dir / "mei"
        self.xml_dir = out_dir / "musicxml"
        self.bekern_dir = out_dir / "bekern"
        self.manifest_path = out_dir / "manifest.jsonl"
        self.processed_path = out_dir / "processed_sources.txt"
        self.summary_path = out_dir / "summary.json"
        self.limit = limit
        self.lock = threading.Lock()
        self.start_time = time.time()

        for d in (self.pages_dir, self.mei_dir, self.xml_dir, self.bekern_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.seen_hashes: Set[str] = set()
        self.processed_sources: Set[str] = set()
        self.count = 0
        self.initial_count = 0
        self.by_institution: Dict[str, int] = {}
        self.by_country: Dict[str, int] = {}
        self.by_license: Dict[str, int] = {}
        self.portrait_count = 0
        self.landscape_count = 0

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
            for line in self.manifest_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not Path(rec.get("image_path", "")).is_file():
                    continue
                if rec.get("image_sha256"):
                    self.seen_hashes.add(rec["image_sha256"])
                if rec.get("rism_id"):
                    self.processed_sources.add(rec["rism_id"])
                self.count += 1
                sig = rec.get("siglum", "unknown")
                ctry = rec.get("country", "unknown")
                lic = rec.get("license_normalized", "unknown")
                self.by_institution[sig] = self.by_institution.get(sig, 0) + 1
                self.by_country[ctry] = self.by_country.get(ctry, 0) + 1
                self.by_license[lic] = self.by_license.get(lic, 0) + 1
                if rec.get("height", 0) >= rec.get("width", 0):
                    self.portrait_count += 1
                else:
                    self.landscape_count += 1
        elif not resume:
            for p in (self.manifest_path, self.processed_path):
                if p.exists():
                    p.unlink()

        if resume and self.processed_path.is_file():
            for line in self.processed_path.read_text(encoding="utf-8").splitlines():
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

    def bump(self, field: Optional[str], reason: str = "") -> None:
        with self.lock:
            if field:
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
        mei_xml: str,
        musicxml: str,
        bekern_txt: str,
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
            img_path = self.pages_dir / f"{safe_sig}_{rism_id}_c{canvas_idx:02d}.jpg"
            mei_path = self.mei_dir / f"{safe_sig}_{rism_id}.mei"
            xml_path = self.xml_dir / f"{safe_sig}_{rism_id}.musicxml"
            bk_path = self.bekern_dir / f"{safe_sig}_{rism_id}-bekern.txt"

            atomic_write_bytes(img_path, jpg_bytes)
            atomic_write_text(mei_path, mei_xml)
            atomic_write_text(xml_path, musicxml)
            atomic_write_text(bk_path, bekern_txt + "\n")

            inst_meta = INTL_INSTITUTIONS.get(siglum, {"institution": siglum, "country": siglum.split("-")[0], "archive": "e-manuscripta.ch"})
            rec = dict(record_meta)
            rec.update(
                {
                    "rism_id": rism_id,
                    "siglum": siglum,
                    "institution": inst_meta["institution"],
                    "country": inst_meta["country"],
                    "archive": inst_meta["archive"],
                    "canvas_index": canvas_idx,
                    "image_path": str(img_path),
                    "mei_path": str(mei_path),
                    "musicxml_path": str(xml_path),
                    "bekern_path": str(bk_path),
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
            ctry = inst_meta["country"]
            self.by_country[ctry] = self.by_country.get(ctry, 0) + 1
            lic = rec.get("license_normalized", "unknown")
            self.by_license[lic] = self.by_license.get(lic, 0) + 1
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
            summary = {
                "total_verified_pairs": self.count,
                "session_new_pairs": new_pairs,
                "limit": self.limit,
                "by_country": dict(sorted(self.by_country.items())),
                "by_institution": dict(sorted(self.by_institution.items())),
                "by_license": dict(sorted(self.by_license.items())),
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
            }
            atomic_write_text(self.summary_path, json.dumps(summary, indent=2, ensure_ascii=False))
            return summary


def process_intl_source(
    source_url: str,
    siglum: str,
    state: IntlHarvestState,
    assembly_bin: Path,
    eval_slugs: Set[str],
    max_dim: int,
    clip_threshold: float,
) -> Optional[str]:
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
        src_text = r_src.text
        try:
            src_json = r_src.json()
        except Exception:
            state.bump("no_iiif_url", "rism_bad_json")
            return None
        finally:
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

        # 1. Extract allowlisted e-manuscripta.ch IIIF manifest URL
        manifest_url, doi_url, pf_html, pf_url = extract_emanuscripta_manifest(session, src_text)
        if not manifest_url:
            state.bump("no_iiif_url", "no_allowed_iiif")
            return None

        # 2. Fetch IIIF manifest & strictly verify per-item PDM 1.0 / CC0 / CC-BY 4.0 license
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

        lic_ok, lic_norm, lic_uri, landing_url = verify_emanuscripta_license(
            session, manifest, manifest_url, doi_url=doi_url, prefetched_html=pf_html, prefetched_url=pf_url
        )
        if not lic_ok:
            state.bump("license_rejected", lic_norm)
            return None
        state.bump("license_verified")

        # 3. Convert first valid PAE incipit -> MEI -> MusicXML -> post-I1 bekern + bekern-decode rc=0
        incipits = src_json.get("incipits", {}).get("items", [])
        if not isinstance(incipits, list) or not incipits:
            state.bump("pae_attempted", "no_incipits")
            return None

        art: Optional[Dict[str, Any]] = None
        last_err = ""
        state.bump("pae_attempted")
        for inc in incipits[:3]:
            res, err = convert_incipit_full(session, inc, assembly_bin)
            if res is not None:
                art = res
                break
            last_err = err

        if art is None:
            state.bump(None, last_err or "pae_failed")
            return None
        state.bump("pae_succeeded")

        # 4. Find first music canvas with CPU CLIP & download high-res image (<= 1800 px)
        canvases = extract_canvases(manifest)
        hires_img, canvas, img_url, clip_stats = find_first_intl_music_canvas(
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
            "landing_page_url": landing_url,
            "image_url": img_url,
            "canvas_label": canvas.get("label", ""),
            "total_canvases": len(canvases),
            "license_verified": True,
            "license_normalized": lic_norm,
            "license_uri": lic_uri,
            "clip_stats": clip_stats,
            "n_bars": art["n_bars"],
            "n_tokens": art["n_tokens"],
            "bekern_decode_rc": art["bekern_decode_rc"],
            "incipit_id": art["incipit_id"],
            "mei_url": art["mei_url"],
            "pae_clef": art["pae_clef"],
            "pae_keysig": art["pae_keysig"],
            "pae_timesig": art["pae_timesig"],
            "pae_notation": art["pae_notation"],
            "bekern": art["bekern"],
        }
        saved = state.record_pair(
            rism_id=rism_id,
            siglum=siglum,
            canvas_idx=canvas["index"],
            jpg_bytes=jpg_bytes,
            width=w,
            height=h,
            mei_xml=art["mei_xml"],
            musicxml=art["musicxml"],
            bekern_txt=art["bekern"],
            record_meta=record_meta,
        )
        return rism_id if saved else None
    finally:
        state.mark_processed(rism_id)
        session.close()


def fetch_intl_source_urls(siglum: str, fq_dig: str, max_pages: int = 10, rows: int = 100) -> List[str]:
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


def write_intl_usage_md(out_dir: Path, summary: Dict[str, Any], part_a_added: List[Dict[str, Any]]) -> Path:
    total = summary["total_verified_pairs"]
    port = summary["portrait_count"]
    land = summary["landscape_count"]
    port_pct = (100.0 * port / total) if total else 0.0
    land_pct = (100.0 * land / total) if total else 0.0

    inst_rows = "\n".join(
        f"| `{sig}` | {INTL_INSTITUTIONS.get(sig, {}).get('institution', sig)} | "
        f"{INTL_INSTITUTIONS.get(sig, {}).get('country', sig.split('-')[0])} | "
        f"{INTL_INSTITUTIONS.get(sig, {}).get('archive', 'e-manuscripta.ch')} | {cnt} |"
        for sig, cnt in sorted(summary.get("by_institution", {}).items())
    )
    lic_rows = "\n".join(
        f"- `{lic}`: **{cnt}** pairs (every item verified via per-item license URI on landing page)"
        for lic, cnt in sorted(summary.get("by_license", {}).items())
    )

    md = f"""# International Labeled Handwritten Music Corpus (`D:\\tmp\\rism-pdm-hw-intl\\`)

Generated reproducibly by `tmp-train/mvd/harvest_intl_labeled_hw.py` (CPU-only, strictly commercial-safe).

## 1. Corpus Summary

- **Total verified `(page_image, MEI, MusicXML, post-I1 bekern)` pairs:** **{total}**
- **Orientation:** **{port}** portrait ({port_pct:.1f}%) / **{land}** landscape ({land_pct:.1f}%)
- **`bekern-decode` pass rate:** **100.0%** ({total}/{total} `rc == 0`, post-I1 canonical)
- **Eval leakage check:** **0** overlap with `data-gen/score-corpus/suite31/suite31_hw_manifest.json`
- **Part A (OpenScore Lieder full-work matches on `e-manuscripta.ch`):** {len(part_a_added)} new full-work matches added to `D:\\tmp\\real-hw-corpus\\train_works.jsonl`

| RISM Siglum | Institution | Country | Archive | Verified Pairs |
|---|---|---|---|---:|
{inst_rows}

## 2. Commercial-Safe License Provenance

{lic_rows}
- **RISM PAE/MEI incipits:** `CC-BY 4.0` (RISM Online).
- **Zero NC / InC / Gallica / unverified-license data:**
  - `iiif.rism.digital` (`CH-Zz`, `CH-E`, `CH-BEl`, `CH-Lz`, `B-Bc`) rejected (no license metadata in IIIF manifests).
  - `iiif.bodleian.ox.ac.uk` (`GB-Ob`) rejected (`CC BY-NC 4.0`).
  - `archive.org` (`GB-WCr`) rejected (`CC BY-NC-ND 4.0`).
  - `alvin-portal.org` (`S-Uu`, `S-L`) skipped (Anubis anti-bot PoW gate).
  - `bdh-rd.bne.es` (`E-Mn`) skipped (HTTP 403 WAF).

## 3. Directory Layout

- `pages/{{siglum}}_{{rism_id}}_c{{idx:02d}}.jpg` — high-res manuscript music page (`<= 1800 px`, RGB JPEG, CPU-CLIP filtered).
- `mei/{{siglum}}_{{rism_id}}.mei` — patched MEI incipit XML (`<keySig>`/`<meterSig>` lifted to `<scoreDef>`).
- `musicxml/{{siglum}}_{{rism_id}}.musicxml` — MusicXML converted from MEI via `music21`.
- `bekern/{{siglum}}_{{rism_id}}-bekern.txt` — canonical post-I1 `bekern` target (`assembly-serialize bekern` + `bekern-decode rc == 0`).
- `manifest.jsonl` — per-pair metadata (`rism_id`, `siglum`, `institution`, `country`, `archive`, `license_uri`, `orientation`, `width`, `height`, `n_bars`, `n_tokens`, `bekern_decode_rc`, file paths).
- `summary.json` — aggregate harvest statistics.

## 4. Intended Training Uses

1. **Stage-1 Equivariance & Self-Supervised Pretraining:** Real international handwritten manuscript pages (`pages/*.jpg`) for geometric/photometric equivariance and staff-line U-Net generalization across Swiss/Central-European historical hands.
2. **Stage-2 Opening-Staff Incipit Alignment:** Each page is paired with a human-transcribed RISM opening incipit (`bekern/*-bekern.txt`, `n_bars` measures, `n_tokens` tokens) for opening-staff OMR supervision and forced-alignment training.
3. **Per-Scribe / Per-Archive Style Bank:** Distinct historical calligraphy, ink bleed, paper tones, and staff-ruling styles from Universitätsbibliothek Basel (`CH-Bu`) and Zentralbibliothek Solothurn (`CH-SO`) to diversify handwritten OMR robustness beyond German/US/Polish/Austrian archives.
"""
    usage_p = out_dir / "USAGE.md"
    atomic_write_text(usage_p, md)
    return usage_p


def verify_zero_eval_overlap(manifest_path: Path) -> Tuple[int, int]:
    """Assert 0 overlap with suite31_hw_manifest.json and 100% bekern-decode rc=0."""
    ev_data = json.loads(SUITE31_HW_MANIFEST.read_text(encoding="utf-8"))
    eval_slugs: Set[str] = set()
    for p in ev_data.get("pieces", []):
        for k in ("slug", "work_id"):
            v = str(p.get(k, "")).strip().lower()
            if v:
                eval_slugs.add(v)
                if v.startswith("hw-"):
                    eval_slugs.add(v[3:])

    total = 0
    overlaps = 0
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        total += 1
        assert rec.get("bekern_decode_rc") == 0, f"Non-zero bekern_decode_rc in {rec.get('rism_id')}"
        assert rec.get("license_verified") is True and rec.get("license_uri"), f"Missing license proof in {rec.get('rism_id')}"
        rid = str(rec.get("rism_id", "")).lower()
        title_norm = normalize_text(rec.get("title", ""))
        comp_norm = normalize_text(rec.get("creator", ""))
        if rid in eval_slugs or is_held_out_hw(comp_norm, title_norm):
            overlaps += 1
    assert overlaps == 0, f"Found {overlaps} eval overlaps in {manifest_path}!"
    return total, overlaps


def main() -> int:
    ap = argparse.ArgumentParser(description="Harvest commercial-safe international labeled HW manuscripts (CPU-only).")
    ap.add_argument("--limit", type=int, default=300, help="Target number of verified Part-B pairs.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help="Output directory (never D:\\tmp\\rism-pdm-hw).")
    ap.add_argument("--lieder-dir", type=Path, default=DEFAULT_LIEDER_DIR, help="OpenScore Lieder CC0 scores directory.")
    ap.add_argument("--workers", type=int, default=6, help="CPU worker threads.")
    ap.add_argument("--max-dim", type=int, default=1800, help="Max JPEG dimension (px).")
    ap.add_argument("--clip-threshold", type=float, default=CLIP_MUSIC_THRESHOLD, help="CPU CLIP music threshold.")
    ap.add_argument("--max-search-pages", type=int, default=12, help="Max RISM search pages (100/page) per query.")
    ap.add_argument("--assembly-bin", type=Path, default=DEFAULT_ASSEMBLY_BIN, help="Path to assembly-serialize.exe.")
    ap.add_argument("--resume", action="store_true", help="Resume from existing manifest and processed_sources.")
    ap.add_argument("--skip-part-a", action="store_true", help="Skip Part A OpenScore Lieder e-manuscripta matching.")
    ap.add_argument("--skip-part-b", action="store_true", help="Skip Part B RISM incipit harvesting.")
    args = ap.parse_args()

    # Safety guard: never touch D:\tmp\rism-pdm-hw
    if args.out.resolve() == Path(r"D:\tmp\rism-pdm-hw").resolve():
        print("ERROR: Refusing to write into D:\\tmp\\rism-pdm-hw (reserved for another task).", file=sys.stderr)
        return 1

    if not args.assembly_bin.is_file():
        print(f"ERROR: assembly-serialize.exe not found at {args.assembly_bin}", file=sys.stderr)
        return 1

    eval_slugs = load_eval_slugs()

    # Pre-warm CPU CLIP model once on main thread
    t0_clip = time.time()
    get_clip_classifier()
    print(f"[clip] CPU CLIP classifier ready in {time.time() - t0_clip:.1f}s", flush=True)

    part_a_added: List[Dict[str, Any]] = []
    if not args.skip_part_a:
        part_a_added = run_part_a_openscore(args.out, args.lieder_dir, args.assembly_bin, eval_slugs)

    if args.skip_part_b:
        return 0

    state = IntlHarvestState(args.out, limit=args.limit, resume=args.resume)
    print(
        f"[Part B init] out={args.out} limit={args.limit} resume={args.resume} "
        f"existing={state.count} processed={len(state.processed_sources)}",
        flush=True,
    )

    if not state.is_done():
        # Queue order: CH-Bu has-iiif:true (fastest, 213 direct IIIF) + CH-SO has-digitization:true (75) + CH-Bu has-digitization:true (1,572 via DOI->IIIF)
        seen_urls: Set[str] = set()
        work_queue: List[Tuple[str, str]] = []
        search_plan = [
            ("CH-Bu", "has-iiif:true", 3),
            ("CH-SO", "has-digitization:true", 2),
            ("CH-Bu", "has-digitization:true", args.max_search_pages),
        ]
        for sig, fq_dig, mpages in search_plan:
            urls = fetch_intl_source_urls(sig, fq_dig, max_pages=mpages, rows=100)
            added_q = 0
            for u in urls:
                rid = u.rstrip("/").split("/")[-1]
                if u in seen_urls or rid in state.processed_sources:
                    continue
                seen_urls.add(u)
                work_queue.append((u, sig))
                added_q += 1
            print(f"[search] {sig} ({fq_dig}): queued {added_q} unprocessed manuscript candidates", flush=True)

        print(f"[harvest] starting with {len(work_queue)} candidates across {args.workers} CPU workers", flush=True)

        last_log = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            in_flight: Set[concurrent.futures.Future[Optional[str]]] = set()
            idx = 0
            max_in_flight = args.workers * 3

            while (idx < len(work_queue) or in_flight) and not state.is_done():
                while idx < len(work_queue) and len(in_flight) < max_in_flight and not state.is_done():
                    src_url, sig = work_queue[idx]
                    idx += 1
                    fut = ex.submit(
                        process_intl_source,
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
                        state.bump(None, f"worker_exc:{type(e).__name__}")

                now = time.time()
                if now - last_log >= 15.0:
                    s = state.write_summary()
                    print(
                        f"[progress] {s['total_verified_pairs']}/{args.limit} pairs "
                        f"({s['pairs_per_min']:.1f}/min) | inst={s['by_institution']} | "
                        f"port={s['portrait_count']} land={s['landscape_count']} | "
                        f"pae={s['pae_succeeded']}/{s['pae_attempted']} ({s['pae_conversion_rate']*100:.1f}%) | "
                        f"lic_ok={s['license_verified']} lic_rej={s['license_rejected']} "
                        f"no_music={s['no_music_canvas']}",
                        flush=True,
                    )
                    last_log = now

            for fut in in_flight:
                fut.cancel()

    summary = state.write_summary()
    usage_p = write_intl_usage_md(args.out, summary, part_a_added)
    total_ver, overlaps = verify_zero_eval_overlap(state.manifest_path)
    print(f"[verify] manifest={state.manifest_path} total={total_ver} eval_overlaps={overlaps} usage={usage_p}", flush=True)
    print("[finished] summary:\n" + json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
