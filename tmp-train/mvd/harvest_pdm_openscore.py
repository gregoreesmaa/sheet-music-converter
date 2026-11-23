#!/usr/bin/env python3
"""Route-B Commercial-Safe Harvester: PDM 1.0 IIIF Full Manuscripts x CC0 OpenScore.

Searches RISM Online for real handwritten (autograph or manuscript copy) digitized
manuscripts held at commercial-safe IIIF libraries:
  - SBB Berlin (D-B, digital.staatsbibliothek-berlin.de)
  - SLUB Dresden (D-Dl, digital.slub-dresden.de)
  - ONB Vienna (A-Wn, onb.ac.at -- strictly rejecting NoC-NC / missing PDM)
  - Polona (PL-Wn, polona.pl)
  - Library of Congress (US-Wc, loc.gov)

STRICTLY NEVER downloads BnF Gallica or any NC / InC / unverified source.
Every IIIF manifest is fetched and verified for explicit PDM 1.0 / CC0 / US Public
Domain rights before any page image is downloaded.

Pairs each verified handwritten manuscript with its exact human-engraved CC0
OpenScore Lieder MusicXML ground truth (extracted from `.mxl` into `xml/{slug}.musicxml`).

CPU + network only (`CUDA_VISIBLE_DEVICES=""`). Resumable via `manifest.jsonl`.

Usage:
    .\\.venv\\Scripts\\python.exe tmp-train/mvd/harvest_pdm_openscore.py ^
        --lieder-dir D:\\tmp\\openscore-lieder\\scores ^
        --out D:\\tmp\\pdm-openscore-hw ^
        --max-works 30 --resume
"""

from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import os

# Hard-disable CUDA/GPU before anything else (CPU + network only)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import re
import sys
import time
import unicodedata
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import urlparse

import requests
from PIL import Image

USER_AGENT = "sheet-music-converter-research/1.0 (commercial-safe-pdm-iiif-harvester)"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LIEDER_DIR = Path(r"D:\tmp\openscore-lieder\scores")
DEFAULT_OUT_DIR = Path(r"D:\tmp\pdm-openscore-hw")
DEFAULT_GATE20_MANIFEST = REPO_ROOT / "data-gen" / "score-corpus" / "gate20" / "gate_manifest.json"
DEFAULT_SUITE31_MANIFEST = REPO_ROOT / "data-gen" / "score-corpus" / "suite31" / "suite31_manifest.json"

# Allowed RISM holding institution sigla -> canonical library name
ALLOWED_LIBRARIES: Dict[str, str] = {
    "D-B": "SBB Berlin",
    "D-Dl": "SLUB Dresden",
    "A-Wn": "ONB Vienna",
    "PL-Wn": "Polona",
    "US-Wc": "Library of Congress",
}

# Allowed IIIF manifest / image host suffixes (hard allowlist)
ALLOWED_HOST_SUFFIXES: Tuple[str, ...] = (
    "staatsbibliothek-berlin.de",
    "slub-dresden.de",
    "onb.ac.at",
    "polona.pl",
    "bn.org.pl",
    "loc.gov",
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
    "rightsstatements.org/vocab/inc",
    "in copyright",
    "all rights reserved",
    "urheberrechtsschutz",
)

# Explicit commercial-safe license markers (must match at least one)
PDM_MARKERS: Tuple[Tuple[str, str], ...] = (
    ("creativecommons.org/publicdomain/mark/1.0", "PDM 1.0"),
    ("public domain mark 1.0", "PDM 1.0"),
    ("pdm 1.0", "PDM 1.0"),
    ("creativecommons.org/publicdomain/zero/1.0", "CC0 1.0"),
    ("cc0 1.0", "CC0 1.0"),
    ("publicdomain/zero", "CC0 1.0"),
    ("rightsstatements.org/vocab/noc-us", "US Public Domain"),
    ("no known copyright", "US Public Domain"),
    ("us public domain", "US Public Domain"),
)

# Held-out suite31_hw works (strictly excluded everywhere)
HELD_OUT_HW_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("beethoven", "135"),
    ("mozart", "464"),
    ("mozart", "387"),
    ("mozart", "421"),
    ("schubert", "810"),
    ("schubert", "death and the maiden"),
    ("schubert", "tod und das madchen"),
)

# Canvas labels that indicate archival covers, bindings, color bars, or blank flyleaves
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
)

# Target OpenScore Lieder composers prioritized by rich SBB/SLUB/ONB manuscript holdings
# Tuple: (OpenScore folder name, RISM query string, required surname token, required given token or "")
TARGET_COMPOSERS: Tuple[Tuple[str, str, str, str], ...] = (
    ("Schubert,_Franz", "Schubert", "schubert", "franz"),
    ("Hensel,_Fanny", "Hensel", "hensel", "fanny"),
    ("Schumann,_Robert", "Schumann", "schumann", "robert"),
    ("Schumann,_Clara", "Clara Schumann", "schumann", "clara"),
    ("Brahms,_Johannes", "Brahms", "brahms", "johannes"),
    ("Mendelssohn,_Felix", "Mendelssohn", "mendelssohn", "felix"),
    ("Beethoven,_Ludwig_van", "Beethoven", "beethoven", "ludwig"),
    ("Reichardt,_Louise", "Reichardt", "reichardt", "louise"),
    ("Wolf,_Hugo", "Hugo Wolf", "wolf", "hugo"),
    ("Franz,_Robert", "Robert Franz", "franz", "robert"),
    ("Cornelius,_Peter", "Peter Cornelius", "cornelius", "peter"),
    ("Schröter,_Corona", "Corona Schröter", "schroter", "corona"),
    ("Zumsteeg,_Emilie", "Zumsteeg", "zumsteeg", "emilie"),
    ("Mahler,_Gustav", "Mahler", "mahler", "gustav"),
    ("Mozart,_Wolfgang_Amadeus", "Mozart", "mozart", "wolfgang"),
    ("Lang,_Josephine", "Josephine Lang", "lang", "josephine"),
)


def normalize_text(text: str) -> str:
    """Lowercase, strip German umlauts/diacritics cleanly, collapse to ascii tokens."""
    s = (text or "").replace("ß", "ss").replace("ẞ", "ss")
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
    if not url:
        return False
    low = url.lower()
    for bad in FORBIDDEN_MARKERS:
        if bad in low:
            return False
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return False
    return any(host == suf or host.endswith("." + suf) for suf in ALLOWED_HOST_SUFFIXES)


def verify_iiif_rights(manifest: Dict[str, Any], manifest_url: str) -> Tuple[bool, str]:
    """Strictly verify that an IIIF manifest declares a commercial-safe PD/CC0 license.

    Returns `(is_safe, normalized_rights_or_reason)`.
    """
    if not is_allowed_url(manifest_url):
        return False, "forbidden_host"

    chunks: List[str] = []
    for key in ("rights", "license", "attribution", "requiredStatement"):
        val = manifest.get(key)
        if val:
            chunks.append(json.dumps(val, ensure_ascii=False) if not isinstance(val, str) else val)

    for item in manifest.get("metadata", []) or []:
        if isinstance(item, dict):
            lbl = str(item.get("label", "")).lower()
            if any(k in lbl for k in ("lizenz", "licen", "recht", "right", "copyright", "nutzung")):
                chunks.append(json.dumps(item, ensure_ascii=False))

    combined = " ".join(chunks).lower()
    # Hard reject if any NC / InC / Gallica marker appears anywhere in rights fields
    for bad in FORBIDDEN_MARKERS:
        if bad in combined:
            return False, f"rejected_marker:{bad}"

    # Require an explicit commercial-safe PD/CC0 marker
    for marker, canonical in PDM_MARKERS:
        if marker in combined:
            return True, canonical

    return False, "no_verified_pdm_marker"


@dataclass
class OpenScoreSong:
    composer_folder: str
    composer_display: str
    surname: str
    given: str
    collection: str
    raw_title: str
    clean_title: str
    norm_title: str
    mxl_path: Path
    catalog_tokens: Set[str] = field(default_factory=set)
    eval_overlap: bool = False


def clean_openscore_title(raw: str) -> str:
    """Strip leading numbering like '01_' or '2_' and underscores from OpenScore song folder names."""
    t = raw.replace("_", " ").strip()
    t = re.sub(r"^\d+[a-z]?\s*[-._)]?\s+", "", t)
    # Remove trailing catalog tags like ", MWV J 5" or ", D.877" for title matching
    t_no_cat = re.sub(r",\s*(?:op\.?|d\.?|mwv|woo|k\.?|kv|bwv|hob\.?)\s*[0-9a-z.\s]+$", "", t, flags=re.I)
    return t_no_cat.strip() or t


def extract_catalog_tokens(text: str) -> Set[str]:
    """Extract normalized catalog numbers like 'd 877', 'op 48', 'mwv j 5'."""
    out: Set[str] = set()
    for m in re.finditer(r"\b(d|op|mwv|woo|kv|k|bwv)\.?\s*([0-9]+[a-z]?)\b", text, flags=re.I):
        out.add(f"{m.group(1).lower()} {m.group(2).lower()}")
    return out


def load_eval_titles() -> Set[str]:
    """Load normalized titles/slugs from gate20 and suite31 manifests to flag eval overlap."""
    out: Set[str] = set()
    for p in (DEFAULT_GATE20_MANIFEST, DEFAULT_SUITE31_MANIFEST):
        if p.is_file():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                for item in data.get("pages", []) + data.get("pieces", []):
                    for k in ("id", "score", "slug", "title"):
                        v = item.get(k)
                        if v:
                            nt = normalize_text(str(v))
                            if len(nt) >= 5:
                                out.add(nt)
            except Exception:
                pass
    return out


def index_openscore_lieder(lieder_dir: Path) -> Dict[str, List[OpenScoreSong]]:
    """Index all CC0 OpenScore Lieder `.mxl` files grouped by composer folder."""
    eval_titles = load_eval_titles()
    by_composer: Dict[str, List[OpenScoreSong]] = {}
    for folder, _q, surname, given in TARGET_COMPOSERS:
        cdir = lieder_dir / folder
        if not cdir.is_dir():
            continue
        songs: List[OpenScoreSong] = []
        composer_display = folder.replace(",_", ", ").replace("_", " ")
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
            # Check eval overlap
            ov = False
            for et in eval_titles:
                if norm_t in et or et in norm_t:
                    ov = True
                    break
            # Also handle parenthetical alternate titles e.g. "Lied der Mignon (Heiss mich nicht reden)"
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
        by_composer[folder] = songs
    return by_composer


def title_variants(song: OpenScoreSong) -> List[str]:
    """Generate normalized title variants (including parenthetical subtitle if present)."""
    variants = [song.norm_title]
    # Parenthetical e.g. "Lied der Mignon (Heiss mich nicht reden)" -> also "heiss mich nicht reden"
    for m in re.finditer(r"\(([^)]{4,})\)", song.clean_title):
        nt = normalize_text(m.group(1))
        if len(nt) >= 5 and nt not in variants:
            variants.append(nt)
    # Without parenthetical
    no_paren = normalize_text(re.sub(r"\([^)]*\)", " ", song.clean_title))
    if len(no_paren) >= 5 and no_paren not in variants:
        variants.append(no_paren)
    return variants


def is_held_out_hw(composer_norm: str, title_norm: str) -> bool:
    for comp, tok in HELD_OUT_HW_PAIRS:
        if comp in composer_norm and re.search(rf"\b{re.escape(tok)}\b", title_norm):
            return True
    return False


def rism_search_composer(
    session: requests.Session,
    query: str,
    lib_siglum: str,
    max_pages: int = 8,
    rows: int = 100,
) -> List[Dict[str, Any]]:
    """Fetch digitized manuscript records from RISM Online for a composer + library."""
    items: List[Dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        params = [
            ("q", query),
            ("mode", "sources"),
            ("rows", str(rows)),
            ("page", str(page)),
            ("fq", "has-digitization:true"),
            ("fq", "source-type:manuscript"),
            ("fq", f"holding-institution:{lib_siglum}"),
            ("fq", "hide-composite-volumes:true"),
        ]
        try:
            r = session.get(
                "https://rism.online/search",
                params=params,
                headers={"Accept": "application/ld+json"},
                timeout=25,
            )
            if not r.ok:
                break
            data = r.json()
        except Exception:
            break
        batch = data.get("items", []) or []
        if not batch:
            break
        items.extend(batch)
        total = int(data.get("totalItems", 0) or 0)
        if len(items) >= total or len(batch) < rows:
            break
    return items


def match_rism_to_openscore(
    rism_item: Dict[str, Any],
    songs: Sequence[OpenScoreSong],
    req_surname: str,
    req_given: str,
) -> Optional[Tuple[OpenScoreSong, float, str]]:
    """Match a RISM search item against local OpenScore songs by the same composer.

    Returns `(song, score, match_reason)` or None.
    """
    summary = rism_item.get("summary") or {}
    comp_vals = ((summary.get("sourceComposer") or {}).get("value") or {}).get("none") or []
    rism_comp_raw = " ".join(comp_vals)
    rism_comp_norm = normalize_text(rism_comp_raw)

    # Strict composer verification (avoid matching works merely owned/copied by the composer)
    if req_surname not in rism_comp_norm:
        return None
    if req_given and req_given not in rism_comp_norm:
        # Handle "Mendelssohn Bartholdy, Felix" or "Hensel, Fanny"
        if not (req_surname == "hensel" and "mendelssohn" in rism_comp_norm and "fanny" in rism_comp_norm):
            return None

    label_en = ((rism_item.get("label") or {}).get("en") or [""])[0]
    if not label_en:
        return None

    # Exclude sketches, fragments, arrangements, excerpts so pages match full MusicXML
    low_lbl = label_en.lower()
    if any(w in low_lbl for w in ("(sketches)", "(fragment", "(arrangement)", "(excerpts)")):
        return None

    # Extract RISM work title (text before first ';' and before '–<key>')
    rism_title_part = label_en.split(";")[0]
    rism_title_clean = re.split(r"[–—]", rism_title_part)[0].strip()
    rism_title_norm = normalize_text(rism_title_clean)
    if len(rism_title_norm) < 4:
        return None

    if is_held_out_hw(rism_comp_norm, normalize_text(label_en)):
        return None

    # Reject generic non-song genres (Symphonies, Masses, Concertos, Operas)
    if any(
        rism_title_norm.startswith(g)
        for g in ("masses", "symphonies", "concertos", "sonatas", "overtures", "trios", "quartets", "quintets", "preludes", "fugues", "etudes")
    ):
        return None

    rism_cats = extract_catalog_tokens(label_en)

    best: Optional[Tuple[OpenScoreSong, float, str]] = None
    for song in songs:
        for v in title_variants(song):
            if len(v) < 4:
                continue
            # 1. Exact normalized title match
            if v == rism_title_norm:
                score = 1.0
                if song.catalog_tokens and (song.catalog_tokens & rism_cats):
                    score = 1.2
                if best is None or score > best[1]:
                    best = (song, score, "exact_title")
            # 2. Multi-word containment (min 2 words and >= 10 chars to avoid false positives)
            elif len(v) >= 10 and " " in v and (
                re.search(rf"\b{re.escape(v)}\b", rism_title_norm)
                or (len(rism_title_norm) >= 10 and " " in rism_title_norm and re.search(rf"\b{re.escape(rism_title_norm)}\b", v))
            ):
                score = 0.85
                if song.catalog_tokens and (song.catalog_tokens & rism_cats):
                    score = 1.05
                if best is None or score > best[1]:
                    best = (song, score, "phrase_match")

    return best


def resolve_iiif_manifest_url(
    session: requests.Session,
    rism_source_url: str,
    lib_siglum: str,
) -> Optional[str]:
    """Fetch a RISM source JSON-LD record and resolve its commercial-safe IIIF manifest URL."""
    try:
        r = session.get(rism_source_url, headers={"Accept": "application/ld+json"}, timeout=20)
        if not r.ok:
            return None
        txt = r.text
    except Exception:
        return None

    urls = sorted(set(re.findall(r'https?://[^\s"<>\\]+', txt)))

    # 1. Direct IIIF manifest URL in RISM record (e.g., SLUB Dresden)
    for u in urls:
        if ("iiif" in u.lower() and "manifest" in u.lower()) and is_allowed_url(u):
            return u

    # 2. SBB Berlin (D-B): direct PPN or resolver.staatsbibliothek-berlin.de
    if lib_siglum == "D-B":
        m_ppn = re.search(r"PPN([0-9]+X?)", txt)
        if m_ppn:
            return f"https://content.staatsbibliothek-berlin.de/dc/PPN{m_ppn.group(1)}/manifest"
        for u in urls:
            if "resolver.staatsbibliothek-berlin.de" in u or "digital.staatsbibliothek-berlin.de" in u:
                try:
                    rr = session.get(u, allow_redirects=True, timeout=20)
                    m2 = re.search(r"PPN([0-9]+X?)", rr.url)
                    if m2:
                        return f"https://content.staatsbibliothek-berlin.de/dc/PPN{m2.group(1)}/manifest"
                except Exception:
                    continue

    # 3. SLUB Dresden (D-Dl): digital.slub-dresden.de/id{ID}
    if lib_siglum == "D-Dl":
        for u in urls:
            m_slub = re.search(r"digital\.slub-dresden\.de/id([0-9X]+)", u)
            if m_slub:
                return f"https://iiif.slub-dresden.de/iiif/2/{m_slub.group(1)}/manifest.json"

    # 4. ONB Vienna (A-Wn): data.onb.ac.at/dtl/{ID} -> viewer.onb.ac.at/{OBJ} -> api.onb.ac.at/iiif/presentation/v3/manifest/{OBJ}
    if lib_siglum == "A-Wn":
        for u in urls:
            if "data.onb.ac.at/dtl/" in u:
                try:
                    rr = session.get(u, allow_redirects=True, timeout=20)
                    m_obj = re.search(r"viewer\.onb\.ac\.at/([A-Za-z0-9_]+)", rr.url)
                    if m_obj:
                        return f"https://api.onb.ac.at/iiif/presentation/v3/manifest/{m_obj.group(1)}"
                except Exception:
                    continue

    # 5. Polona (PL-Wn)
    if lib_siglum == "PL-Wn":
        for u in urls:
            m_pol = re.search(r"polona\.pl/item/([A-Za-z0-9]+)", u)
            if m_pol:
                return f"https://polona.pl/iiif/item/{m_pol.group(1)}/manifest.json"

    return None


def canvas_label_text(canvas: Dict[str, Any]) -> str:
    lbl = canvas.get("label", "")
    if isinstance(lbl, dict):
        vals: List[str] = []
        for v in lbl.values():
            if isinstance(v, list):
                vals.extend(str(x) for x in v)
            else:
                vals.append(str(v))
        return " ".join(vals)
    if isinstance(lbl, list):
        return " ".join(str(x) for x in lbl)
    return str(lbl or "")


def is_non_music_canvas(label: str) -> bool:
    nl = normalize_text(label)
    if not nl:
        return False
    return any(tok in nl for tok in NON_MUSIC_CANVAS_LABELS)


def extract_iiif_canvases(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract list of `{id, label, width, height, image_url}` from IIIF v2 or v3 manifest."""
    out: List[Dict[str, Any]] = []

    # IIIF Presentation 2.x
    if "sequences" in manifest and manifest["sequences"]:
        canvases = manifest["sequences"][0].get("canvases", []) or []
        for c in canvases:
            cid = str(c.get("@id") or c.get("id") or "")
            lbl = canvas_label_text(c)
            w = int(c.get("width") or 0)
            h = int(c.get("height") or 0)
            img_url = ""
            imgs = c.get("images") or []
            if imgs:
                res = imgs[0].get("resource") or {}
                svc = res.get("service")
                if isinstance(svc, list) and svc:
                    svc = svc[0]
                svc_id = svc.get("@id") or svc.get("id") if isinstance(svc, dict) else None
                if svc_id:
                    svc_id = str(svc_id).rstrip("/")
                    img_url = f"{svc_id}/full/!2048,2048/0/default.jpg"
                else:
                    img_url = str(res.get("@id") or res.get("id") or "")
            if img_url and is_allowed_url(img_url):
                out.append({"id": cid, "label": lbl, "width": w, "height": h, "image_url": img_url, "fallback_url": str((imgs[0].get("resource") or {}).get("@id") or "") if imgs else ""})
        return out

    # IIIF Presentation 3.0
    if "items" in manifest:
        for c in manifest.get("items") or []:
            cid = str(c.get("id") or c.get("@id") or "")
            lbl = canvas_label_text(c)
            w = int(c.get("width") or 0)
            h = int(c.get("height") or 0)
            img_url = ""
            fallback = ""
            try:
                body = c["items"][0]["items"][0]["body"]
                if isinstance(body, list):
                    body = body[0]
                fallback = str(body.get("id") or "")
                svcs = body.get("service") or []
                if isinstance(svcs, dict):
                    svcs = [svcs]
                if svcs and isinstance(svcs[0], dict):
                    svc_id = str(svcs[0].get("id") or svcs[0].get("@id") or "").rstrip("/")
                    if svc_id:
                        img_url = f"{svc_id}/full/!2048,2048/0/default.jpg"
                if not img_url:
                    img_url = fallback
            except Exception:
                pass
            if img_url and is_allowed_url(img_url):
                out.append({"id": cid, "label": lbl, "width": w, "height": h, "image_url": img_url, "fallback_url": fallback})
    return out


def match_structure_canvases(
    manifest: Dict[str, Any],
    all_canvases: Sequence[Dict[str, Any]],
    song: OpenScoreSong,
    min_pages: int,
    max_pages: int,
) -> Optional[Tuple[str, List[Dict[str, Any]]]]:
    """For multi-song autograph albums (e.g. SBB Berlin), extract the exact song canvas range via IIIF `structures`."""
    structs = manifest.get("structures") or []
    if not structs:
        return None
    by_id = {c["id"]: c for c in all_canvases if c.get("id")}
    variants = title_variants(song)

    for st in structs:
        cids = st.get("canvases") or []
        if not cids or not (min_pages <= len(cids) <= max_pages):
            continue
        lbl = canvas_label_text(st)
        lbl_title = re.split(r"[;–—]", lbl)[0].strip()
        lbl_norm = normalize_text(lbl_title)
        if not lbl_norm:
            continue
        matched = False
        for v in variants:
            if v == lbl_norm or (len(v) >= 8 and re.search(rf"\b{re.escape(v)}\b", lbl_norm)):
                matched = True
                break
        if matched:
            sub = [by_id[cid] for cid in cids if cid in by_id]
            if min_pages <= len(sub) <= max_pages:
                return (lbl, sub)
    return None


def extract_mxl_to_musicxml(mxl_path: Path, out_xml_path: Path) -> Path:
    """Unzip a CC0 OpenScore `.mxl` into a clean `.musicxml` file atomically."""
    out_xml_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(mxl_path, "r") as zf:
        # Find rootfile from META-INF/container.xml or largest .xml/.musicxml
        xml_member = None
        if "META-INF/container.xml" in zf.namelist():
            ct = zf.read("META-INF/container.xml").decode("utf-8", errors="ignore")
            m = re.search(r'full-path="([^"]+)"', ct)
            if m and m.group(1) in zf.namelist():
                xml_member = m.group(1)
        if not xml_member:
            cands = [n for n in zf.namelist() if n.lower().endswith((".musicxml", ".xml")) and "meta-inf" not in n.lower()]
            if not cands:
                raise RuntimeError(f"No MusicXML inside {mxl_path}")
            xml_member = max(cands, key=lambda n: zf.getinfo(n).file_size)
        data = zf.read(xml_member)
    tmp = out_xml_path.with_suffix(".musicxml.tmp")
    tmp.write_bytes(data)
    tmp.replace(out_xml_path)
    return out_xml_path


def download_page_image(
    session: requests.Session,
    image_url: str,
    fallback_url: str,
    out_path: Path,
    max_dim: int = 2048,
) -> Optional[Tuple[int, int]]:
    """Download an IIIF page image atomically, ensuring valid RGB JPEG <= max_dim."""
    if out_path.is_file() and out_path.stat().st_size > 4096:
        try:
            with Image.open(out_path) as im:
                return im.size
        except Exception:
            pass

    urls_to_try = [u for u in (image_url, fallback_url) if u and is_allowed_url(u)]
    # Also try IIIF v2 '/full/1600,/0/default.jpg' if '!2048,2048' returns 400
    if "/full/!2048,2048/" in image_url:
        urls_to_try.insert(1, image_url.replace("/full/!2048,2048/", "/full/1600,/"))
        urls_to_try.insert(2, image_url.replace("/full/!2048,2048/", "/full/full/"))

    raw = None
    for u in urls_to_try:
        try:
            r = session.get(u, timeout=30)
            if r.ok and len(r.content) > 4096:
                raw = r.content
                break
        except Exception:
            continue
    if raw is None:
        return None

    try:
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        w, h = im.size
        if max(w, h) > max_dim:
            scale = max_dim / float(max(w, h))
            im = im.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.Resampling.LANCZOS)
            w, h = im.size
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".jpg.tmp")
        im.save(tmp, format="JPEG", quality=92)
        tmp.replace(out_path)
        return (w, h)
    except Exception:
        return None


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


def main() -> int:
    ap = argparse.ArgumentParser(description="Harvest commercial-safe PDM 1.0 IIIF handwritten manuscripts matched to CC0 OpenScore Lieder.")
    ap.add_argument("--lieder-dir", type=Path, default=DEFAULT_LIEDER_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--max-works", type=int, default=30, help="Maximum matched works to download.")
    ap.add_argument("--max-pages-per-work", type=int, default=24, help="Skip giant composite manuscripts exceeding this many music pages.")
    ap.add_argument("--min-pages-per-work", type=int, default=2, help="Minimum music pages required.")
    ap.add_argument("--libraries", nargs="+", default=["D-B", "D-Dl", "A-Wn", "PL-Wn"], choices=list(ALLOWED_LIBRARIES.keys()))
    ap.add_argument("--workers", type=int, default=4, help="Parallel image download workers per work.")
    ap.add_argument("--resume", action="store_true", help="Resume and skip slugs already in manifest.jsonl.")
    ap.add_argument("--dry-run", action="store_true", help="Find and verify matches without downloading page images.")
    args = ap.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"
    existing = load_existing_manifest(manifest_path) if args.resume else {}
    if not args.resume and manifest_path.is_file() and not args.dry_run:
        manifest_path.unlink()

    print(f"[1/4] Indexing CC0 OpenScore Lieder in {args.lieder_dir} ...", flush=True)
    by_composer = index_openscore_lieder(args.lieder_dir)
    total_songs = sum(len(v) for v in by_composer.values())
    print(f"      Indexed {total_songs} CC0 Lieder across {len(by_composer)} target composers.", flush=True)

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    print(f"[2/4] Searching RISM Online digitized manuscripts in {args.libraries} ...", flush=True)
    # Collect matched candidates: keyed by (composer_folder, song.norm_title, rism_id)
    candidates: List[Dict[str, Any]] = []
    seen_song_keys: Set[Tuple[str, str]] = set()

    for folder, q_comp, surname, given in TARGET_COMPOSERS:
        songs = by_composer.get(folder, [])
        if not songs:
            continue
        for lib in args.libraries:
            rism_items = rism_search_composer(session, q_comp, lib, max_pages=6, rows=100)
            matched_here = 0
            for it in rism_items:
                res = match_rism_to_openscore(it, songs, surname, given)
                if not res:
                    continue
                song, score, reason = res
                label_en = ((it.get("label") or {}).get("en") or [""])[0]
                mat_types = [
                    ((m.get("typeLabel") or {}).get("en") or [""])[0]
                    for m in ((it.get("flags") or {}).get("materialTypes") or [])
                ]
                is_autograph = any("autograph" in m.lower() for m in mat_types)
                candidates.append(
                    {
                        "song": song,
                        "score": score + (0.15 if is_autograph else 0.0),
                        "reason": reason,
                        "library_siglum": lib,
                        "library": ALLOWED_LIBRARIES[lib],
                        "rism_url": it.get("id", ""),
                        "rism_label": label_en,
                        "material_type": ", ".join(m for m in mat_types if m) or "Manuscript",
                    }
                )
                matched_here += 1
            if matched_here:
                print(f"      {folder:24s} | {lib:4s} | rism={len(rism_items):3d} -> matched={matched_here}", flush=True)

    # Sort best matches first (autographs + exact matches + non-eval-overlap preferred)
    candidates.sort(key=lambda c: (not c["song"].eval_overlap, c["score"]), reverse=True)
    print(f"[3/4] Resolving IIIF manifests & strictly verifying PDM 1.0/CC0 rights on {len(candidates)} candidates ...", flush=True)

    verified_works: List[Dict[str, Any]] = []
    seen_ranges: Set[Tuple[str, str]] = set()
    manifest_cache: Dict[str, Optional[Dict[str, Any]]] = {}
    # Keep already-downloaded slugs counted
    downloaded_records: Dict[str, Dict[str, Any]] = dict(existing)

    # First resolve and inspect manifests (prioritizing portrait H > W!)
    for cand in candidates:
        song: OpenScoreSong = cand["song"]
        song_key = (song.composer_folder, song.norm_title)
        if song_key in seen_song_keys:
            continue

        mf_url = resolve_iiif_manifest_url(session, cand["rism_url"], cand["library_siglum"])
        if not mf_url or not is_allowed_url(mf_url):
            continue

        if mf_url not in manifest_cache:
            try:
                rm = session.get(mf_url, timeout=25)
                manifest_cache[mf_url] = rm.json() if rm.ok else None
            except Exception:
                manifest_cache[mf_url] = None
        mf_json = manifest_cache[mf_url]
        if not mf_json:
            continue

        ok_rights, rights_str = verify_iiif_rights(mf_json, mf_url)
        if not ok_rights:
            print(f"      [SKIP LICENSE] {cand['library']} | {song.clean_title} -> {rights_str}", flush=True)
            continue

        all_canvases = extract_iiif_canvases(mf_json)
        music_canvases = [c for c in all_canvases if not is_non_music_canvas(c["label"])]
        if not music_canvases:
            music_canvases = all_canvases

        range_key = "FULL"
        if len(music_canvases) > args.max_pages_per_work:
            sub_match = match_structure_canvases(
                mf_json, all_canvases, song, args.min_pages_per_work, args.max_pages_per_work
            )
            if not sub_match:
                continue
            range_key, music_canvases = sub_match
        elif len(music_canvases) < args.min_pages_per_work:
            continue

        if (mf_url, range_key) in seen_ranges:
            continue

        n_pages = len(music_canvases)
        # Determine orientation from canvas dimensions (fallback to portrait if unknown until download)
        p_cnt = sum(1 for c in music_canvases if c["height"] > c["width"] > 0)
        l_cnt = sum(1 for c in music_canvases if c["width"] > c["height"] > 0)
        orient = "portrait" if p_cnt >= l_cnt else "landscape"

        slug = slugify(f"{song.surname}-{song.clean_title}-{cand['library_siglum'].lower()}")
        seen_ranges.add((mf_url, range_key))
        seen_song_keys.add(song_key)

        verified_works.append(
            {
                "slug": slug,
                "cand": cand,
                "mf_url": mf_url,
                "mf_json": mf_json,
                "rights": rights_str,
                "canvases": music_canvases,
                "n_pages": n_pages,
                "orientation": orient,
                "structure_range": range_key,
            }
        )
        print(
            f"      [VERIFIED] {slug:45s} | {cand['library']:12s} | {rights_str:7s} | {n_pages:2d}p | {orient:9s} | {cand['material_type']} | range={range_key[:35]}",
            flush=True,
        )
        if len(verified_works) >= args.max_works * 2:
            break

    # Prioritize portrait H > W works first, then clean non-eval-overlap, then autographs!
    verified_works.sort(
        key=lambda w: (
            w["orientation"] == "portrait",
            not w["cand"]["song"].eval_overlap,
            "autograph" in w["cand"]["material_type"].lower(),
            w["cand"]["score"],
        ),
        reverse=True,
    )
    selected = verified_works[: args.max_works]
    print(f"\n[4/4] Selected {len(selected)} verified PDM works (dry_run={args.dry_run}).", flush=True)

    if args.dry_run:
        p_works = sum(1 for w in selected if w["orientation"] == "portrait")
        l_works = sum(1 for w in selected if w["orientation"] == "landscape")
        print(f"DRY RUN summary: {len(selected)} works ({p_works} portrait, {l_works} landscape).")
        return 0

    for idx, w in enumerate(selected, 1):
        slug = w["slug"]
        if args.resume and slug in downloaded_records:
            print(f"  [{idx}/{len(selected)}] RESUME skip {slug}", flush=True)
            continue

        cand = w["cand"]
        song: OpenScoreSong = cand["song"]

        # Save verified IIIF manifest JSON for audit trail
        mf_out = out_dir / "manifests_iiif" / f"{slug}.json"
        mf_out.parent.mkdir(parents=True, exist_ok=True)
        mf_out.write_text(json.dumps(w["mf_json"], ensure_ascii=False, indent=2), encoding="utf-8")

        # Extract CC0 MusicXML
        xml_out = out_dir / "xml" / f"{slug}.musicxml"
        extract_mxl_to_musicxml(song.mxl_path, xml_out)

        # Download pages in parallel (CPU + network only)
        pages_dir = out_dir / "pages" / slug
        pages_dir.mkdir(parents=True, exist_ok=True)
        canvases = w["canvases"]

        def _dl(i_c: Tuple[int, Dict[str, Any]]) -> Tuple[int, Optional[Path], Optional[Tuple[int, int]]]:
            i, c = i_c
            p_path = pages_dir / f"page_{i:03d}.jpg"
            dims = download_page_image(session, c["image_url"], c.get("fallback_url", ""), p_path)
            return (i, p_path if dims else None, dims)

        page_paths: List[str] = []
        page_dims: List[List[int]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(_dl, enumerate(canvases, 1)))
        results.sort(key=lambda x: x[0])
        for _i, p_path, dims in results:
            if p_path and dims:
                page_paths.append(str(p_path))
                page_dims.append([dims[0], dims[1]])

        if not page_paths:
            print(f"  [{idx}/{len(selected)}] FAILED download for {slug}", flush=True)
            continue

        # Compute exact orientation from actual downloaded pixel dimensions!
        n_port = sum(1 for dw, dh in page_dims if dh > dw)
        n_land = sum(1 for dw, dh in page_dims if dw >= dh)
        actual_orient = "portrait" if n_port >= n_land else "landscape"

        rec = {
            "slug": slug,
            "composer": song.composer_display,
            "title": song.clean_title,
            "library": cand["library"],
            "iiif_manifest": w["mf_url"],
            "rights": w["rights"],
            "n_pages": len(page_paths),
            "orientation": actual_orient,
            "xml_path": str(xml_out),
            "page_paths": page_paths,
            # Additional provenance metadata
            "n_portrait_pages": n_port,
            "n_landscape_pages": n_land,
            "page_dims": page_dims,
            "rism_url": cand["rism_url"],
            "rism_label": cand["rism_label"],
            "material_type": cand["material_type"],
            "structure_range": w.get("structure_range", "FULL"),
            "openscore_mxl": str(song.mxl_path),
            "xml_license": "CC0-1.0 (OpenScore Lieder)",
            "eval_overlap": song.eval_overlap,
        }
        downloaded_records[slug] = rec
        with manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(
            f"  [{idx}/{len(selected)}] SAVED {slug} | {len(page_paths)}p ({n_port}P/{n_land}L) | {actual_orient} | {w['rights']} | {cand['library']}",
            flush=True,
        )

    # Write summary.json
    recs = list(downloaded_records.values())
    total_pages = sum(r["n_pages"] for r in recs)
    total_port_pages = sum(r.get("n_portrait_pages", r["n_pages"] if r["orientation"] == "portrait" else 0) for r in recs)
    total_land_pages = sum(r.get("n_landscape_pages", r["n_pages"] if r["orientation"] == "landscape" else 0) for r in recs)
    by_lib: Dict[str, int] = {}
    by_rights: Dict[str, int] = {}
    for r in recs:
        by_lib[r["library"]] = by_lib.get(r["library"], 0) + 1
        by_rights[r["rights"]] = by_rights.get(r["rights"], 0) + 1
    portrait_holdouts = [
        {"slug": r["slug"], "composer": r["composer"], "title": r["title"], "library": r["library"], "n_pages": r["n_pages"], "rights": r["rights"], "material_type": r.get("material_type", "")}
        for r in recs
        if r["orientation"] == "portrait" and not r.get("eval_overlap", False)
    ]

    summary = {
        "works_matched": len(recs),
        "portrait_works": sum(1 for r in recs if r["orientation"] == "portrait"),
        "landscape_works": sum(1 for r in recs if r["orientation"] == "landscape"),
        "total_pages": total_pages,
        "portrait_pages": total_port_pages,
        "landscape_pages": total_land_pages,
        "by_library": by_lib,
        "by_rights": by_rights,
        "xml_license": "CC0-1.0 (OpenScore Lieder)",
        "portrait_holdout_candidates": portrait_holdouts,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== HARVEST SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
