#!/usr/bin/env python3
"""Bach Chorales (CC-BY 4.0) + NIFC Chopin (CC-BY 4.0) x PDM 1.0 Autograph Harvester.

Harvests two high-precision, 100% commercial-safe labeled handwritten corpora into
D:\\tmp\\pdm-bach-chopin-hw\\:

1. Bach Chorales Figured Bass v2.0 (CC-BY 4.0, Zenodo 5084914) cross-matched by
   BWV number against SBB Berlin (D-B, PDM 1.0) + SLUB Dresden (D-Dl, PDM 1.0)
   handwritten Bach autographs and copyist manuscripts.
2. NIFC Chopin First Editions (CC-BY 4.0, github.com/pl-wnifc/humdrum-chopin-first-editions)
   cross-matched by Op. / WN / KK / KobC catalog against Polona (PL-Wn, polona2.pl
   verified commercial-safe Domena Publiczna) + SBB Berlin (D-B, PDM 1.0) Chopin
   autograph and copyist manuscripts.

STRICT RULES:
  - CPU-only (`CUDA_VISIBLE_DEVICES=""` set before any imports).
  - Never touches `D:\\tmp\\rism-pdm-hw\\`.
  - 100% commercial-safe: per-item PDM 1.0 / Polona commercial-safe Domena Publiczna
    scans x CC-BY 4.0 symbolic scores (zero Gallica, zero NC, zero InC).
  - Converts every matched symbolic score -> MusicXML -> post-I1 canonical `bekern`
    via `assembly-serialize.exe bekern` + validates `bekern-decode` rc=0.
  - Zero eval overlap with `suite31_hw_manifest.json` and `suite31_manifest.json`
    (explicitly excludes `bach-bwv66.6` and `chopin-mazurka06-2` / Chopin Op. 6).
  - Up to `--max-pages-per-work 8` CLIP-verified music pages (`<= 1800 px`, JPEG Q=92).

Usage:
    $env:CUDA_VISIBLE_DEVICES=""
    .\\.venv\\Scripts\\python.exe tmp-train/mvd/harvest_bach_chopin_hw.py `
        --out D:\\tmp\\pdm-bach-chopin-hw --max-pages-per-work 8 --resume
"""

from __future__ import annotations

import os

# Hard-disable CUDA/GPU before anything else (CPU + network only)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import argparse
import concurrent.futures
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

import recanon_i1  # noqa: E402
from harvest_polona_rism import (  # noqa: E402
    extract_polona_item_id,
    is_polona_allowed_url,
    verify_polona_license,
)
from harvest_unlabeled_10k import (  # noqa: E402
    CLIP_MUSIC_THRESHOLD,
    get_clip_classifier,
    is_eval_excluded,
    is_music_page,
    load_eval_slugs,
)

USER_AGENT = "sheet-music-converter-research/1.0 (commercial-safe-bach-chopin-pdm-harvester; CPU-only)"
DEFAULT_OUT_DIR = Path(r"D:\tmp\pdm-bach-chopin-hw")
DEFAULT_BACH_DIR = Path(r"D:\tmp\bach-chorales-fb")
DEFAULT_CHOPIN_DIR = Path(r"D:\tmp\nifc-chopin-repo")
DEFAULT_ASSEMBLY_BIN = (
    REPO_ROOT / "converter" / "assembly-serialize" / "target" / "release" / "assembly-serialize.exe"
)
SUITE31_MANIFEST = REPO_ROOT / "data-gen" / "score-corpus" / "suite31" / "suite31_manifest.json"
SUITE31_HW_MANIFEST = REPO_ROOT / "data-gen" / "score-corpus" / "suite31" / "suite31_hw_manifest.json"
GATE20_MANIFEST = REPO_ROOT / "data-gen" / "score-corpus" / "gate20" / "gate_manifest.json"

ZENODO_BACH_RECORD_ID = "5084914"
CHOPIN_REPO_URL = "https://github.com/pl-wnifc/humdrum-chopin-first-editions.git"

# Hard domain allowlist for manuscript scans/manifests
ALLOWED_HOST_SUFFIXES: Tuple[str, ...] = (
    "staatsbibliothek-berlin.de",
    "slub-dresden.de",
    "polona.pl",
    "polona2.pl",
    "zas.bn.org.pl",
    "bn.org.pl",
)

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

PDM10_MARKERS: Tuple[Tuple[str, str], ...] = (
    ("creativecommons.org/publicdomain/mark/1.0", "PDM 1.0"),
    ("public domain mark 1.0", "PDM 1.0"),
    ("pdm 1.0", "PDM 1.0"),
    ("rightsstatements.org/vocab/pdm/", "PDM 1.0"),
)

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

# Known Chopin title/dedication hints -> canonical Op. key (for Polona manuscripts whose title omits Op.)
CHOPIN_TITLE_HINTS: Tuple[Tuple[str, str], ...] = (
    ("berceuse", "op:57"),
    ("24 preludes", "op:28"),
    ("24. preludes", "op:28"),
    ("jules fontana", "op:40"),
    ("furstenstein", "op:31"),
    ("gutmann", "op:39"),
    ("konneritz", "op:62"),
    ("allegro de concert", "op:46"),
    ("barcarolle", "op:60"),
    ("barkarola", "op:60"),
    ("tarantelle", "op:43"),
    ("tarantela", "op:43"),
    ("bolero", "op:19"),
    ("fantaisie-impromptu", "op:66"),
    ("fantaisie impromptu", "op:66"),
    ("polonaise-fantaisie", "op:61"),
    ("polonaise fantaisie", "op:61"),
    ("polonez-fantazja", "op:61"),
    ("krakowiak", "op:14"),
    ("andante spianato", "op:22"),
    ("variations brillantes", "op:12"),
    ("la ci darem", "op:2"),
    ("hulanka", "op:74/4"),
    ("zyczenie", "op:74/1"),
    ("wiosna", "op:74/2"),
    ("smutna rzeka", "op:74/3"),
    ("gdzie lubi", "op:74/5"),
    ("precz z moich oczu", "op:74/6"),
    ("posel", "op:74/7"),
    ("sliczny chlopiec", "op:74/8"),
    ("melodia", "op:74/9"),
    ("wojaka", "op:74/10"),
    ("dwojaki koniec", "op:74/11"),
    ("moja pieszczotka", "op:74/12"),
    ("nie ma czego trzeba", "op:74/13"),
    ("pierscien", "op:74/14"),
    ("narzeczony", "op:74/15"),
    ("piosnka litewska", "op:74/16"),
    ("spiew z mogily", "op:74/17"),
)

# Per-host politeness semaphores
_HOST_LOCKS: Dict[str, threading.Semaphore] = {}
_HOST_LOCKS_GUARD = threading.Lock()


def get_host_semaphore(url: str) -> threading.Semaphore:
    host = (urlparse(url).hostname or "default").lower()
    with _HOST_LOCKS_GUARD:
        sem = _HOST_LOCKS.get(host)
        if sem is None:
            limit = 4 if "staatsbibliothek-berlin.de" in host else 6
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
    s = s.replace("ł", "l").replace("Ł", "l")
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
    """Strictly verify explicit PDM 1.0 rights on SBB Berlin / SLUB Dresden IIIF manifest."""
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

    combined = " | ".join(chunks)
    low = combined.lower()
    for bad in FORBIDDEN_MARKERS:
        if bad in low:
            return False, f"rejected_marker:{bad}", combined[:240]

    for marker, canonical in PDM10_MARKERS:
        if marker in low:
            return True, canonical, combined[:240]

    return False, "no_verified_pdm10_marker", combined[:240]


def canvas_label_text(canvas: Dict[str, Any]) -> str:
    return " ".join(_flatten_strings(canvas.get("label")))


def is_non_music_canvas(label: str) -> bool:
    nl = normalize_text(label)
    if not nl:
        return False
    return any(tok in nl for tok in NON_MUSIC_CANVAS_LABELS)


def extract_iiif_canvases(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract canvases from IIIF v2 or v3 manifest."""
    out: List[Dict[str, Any]] = []
    if "sequences" in manifest and manifest["sequences"]:
        canvases = manifest["sequences"][0].get("canvases", []) or []
        for idx, c in enumerate(canvases):
            cid = str(c.get("@id") or c.get("id") or "")
            lbl = canvas_label_text(c)
            w = int(c.get("width") or 0)
            h = int(c.get("height") or 0)
            svc_id = ""
            res_id = ""
            imgs = c.get("images") or []
            if imgs:
                res = imgs[0].get("resource") or {}
                res_id = str(res.get("@id") or res.get("id") or "")
                svc = res.get("service")
                if isinstance(svc, list) and svc:
                    svc = svc[0]
                if isinstance(svc, dict):
                    svc_id = str(svc.get("@id") or svc.get("id") or "").rstrip("/")
            if (svc_id and is_allowed_url(svc_id)) or (res_id and is_allowed_url(res_id)):
                out.append(
                    {
                        "index": idx,
                        "id": cid,
                        "label": lbl,
                        "width": w,
                        "height": h,
                        "service_id": svc_id,
                        "resource_id": res_id,
                    }
                )
        return out

    if "items" in manifest:
        for idx, c in enumerate(manifest.get("items") or []):
            cid = str(c.get("id") or c.get("@id") or "")
            lbl = canvas_label_text(c)
            w = int(c.get("width") or 0)
            h = int(c.get("height") or 0)
            svc_id = ""
            res_id = ""
            try:
                body = c["items"][0]["items"][0]["body"]
                if isinstance(body, list):
                    body = body[0]
                res_id = str(body.get("id") or "")
                svcs = body.get("service") or []
                if isinstance(svcs, dict):
                    svcs = [svcs]
                if svcs and isinstance(svcs[0], dict):
                    svc_id = str(svcs[0].get("id") or svcs[0].get("@id") or "").rstrip("/")
            except Exception:
                pass
            if (svc_id and is_allowed_url(svc_id)) or (res_id and is_allowed_url(res_id)):
                out.append(
                    {
                        "index": idx,
                        "id": cid,
                        "label": lbl,
                        "width": w,
                        "height": h,
                        "service_id": svc_id,
                        "resource_id": res_id,
                    }
                )
    return out


def candidate_iiif_image_urls(canvas: Dict[str, Any], max_dim: int = 1800) -> List[str]:
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


def resize_pil_max_dim(im: Image.Image, max_dim: int) -> Image.Image:
    w, h = im.size
    if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        return im.resize(
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            Image.Resampling.LANCZOS,
        )
    return im


def save_jpeg_atomic(im: Image.Image, out_path: Path, quality: int = 92) -> Tuple[int, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".jpg.tmp")
    im.save(tmp, format="JPEG", quality=quality)
    os.replace(tmp, out_path)
    return im.size


def fetch_and_filter_iiif_canvas(
    session: requests.Session,
    canvas: Dict[str, Any],
    max_dim: int = 1800,
    clip_threshold: float = CLIP_MUSIC_THRESHOLD,
) -> Optional[Tuple[Image.Image, str, Dict[str, Any]]]:
    for u in candidate_iiif_image_urls(canvas, max_dim=max_dim):
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
        im = resize_pil_max_dim(im, max_dim=max_dim)
        return im, u, clip_stats
    return None


def fetch_polona_image(session: requests.Session, url: str, timeout: float = 30.0) -> Optional[Image.Image]:
    if not is_polona_allowed_url(url):
        return None
    r = http_get(session, url, headers={"User-Agent": USER_AGENT}, timeout=timeout, retries=2)
    if r is None:
        return None
    try:
        ct = r.headers.get("content-type", "").lower()
        if "image" not in ct and not r.content[:4].startswith((b"\xff\xd8\xff", b"\x89PNG")):
            return None
        im = Image.open(io.BytesIO(r.content)).convert("RGB")
        im.load()
        return im
    except Exception:
        return None
    finally:
        r.close()


_CLIP_LOCK = threading.Lock()


def _clip_check_safe(im: Image.Image, threshold: float) -> Tuple[bool, Dict[str, Any]]:
    with _CLIP_LOCK:
        return is_music_page(im, threshold=threshold)


def collect_polona_music_pages(
    session: requests.Session,
    scans: List[Dict[str, Any]],
    pages_dir: Path,
    max_pages: int = 8,
    max_scan_inspect: int = 24,
    max_dim: int = 1800,
    clip_threshold: float = CLIP_MUSIC_THRESHOLD,
) -> Tuple[List[str], List[List[int]], List[str]]:
    """Collect up to `max_pages` CLIP-verified music pages from a Polona entity (parallel fetch)."""
    page_paths: List[str] = []
    page_dims: List[List[int]] = []
    page_urls: List[str] = []
    if not scans:
        return page_paths, page_dims, page_urls

    cands = [sc for sc in scans[:max_scan_inspect] if isinstance(sc, dict)]

    def _process_scan(idx_sc: Tuple[int, Dict[str, Any]]) -> Optional[Tuple[int, Image.Image, str]]:
        idx, sc = idx_sc
        s_local = requests.Session()
        s_local.headers["User-Agent"] = USER_AGENT
        try:
            thumbs = sc.get("thumbnails") or []
            thumbs_sorted = sorted(
                [t for t in thumbs if isinstance(t, dict) and t.get("url")],
                key=lambda t: int((t.get("dimensions") or {}).get("width") or 0),
                reverse=True,
            )
            preview_img = None
            preview_url = ""
            for th in thumbs_sorted:
                u = str(th["url"])
                preview_img = fetch_polona_image(s_local, u, timeout=20.0)
                if preview_img is not None:
                    preview_url = u
                    break
            if preview_img is None:
                return None

            ok, _stats = _clip_check_safe(preview_img, threshold=clip_threshold)
            if not ok:
                return None

            chosen_img = None
            chosen_url = ""
            for res in sc.get("resources") or []:
                if isinstance(res, dict) and res.get("mime") == "image/jpeg" and res.get("url"):
                    full_url = str(res["url"])
                    hires = fetch_polona_image(s_local, full_url, timeout=35.0)
                    if hires is not None:
                        chosen_img = hires
                        chosen_url = full_url
                        break
            if chosen_img is None and max(preview_img.size) >= 700:
                chosen_img = preview_img
                chosen_url = preview_url
            if chosen_img is None:
                return None
            return (idx, resize_pil_max_dim(chosen_img, max_dim=max_dim), chosen_url)
        finally:
            s_local.close()

    # Process in waves of 8 scans so we stop early once we have max_pages
    collected: List[Tuple[int, Image.Image, str]] = []
    for start in range(0, len(cands), 8):
        if len(collected) >= max_pages:
            break
        batch = list(enumerate(cands[start : start + 8], start))
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
            for res in ex.map(_process_scan, batch):
                if res is not None:
                    collected.append(res)

    collected.sort(key=lambda x: x[0])
    for p_idx, (_orig_i, im, url) in enumerate(collected[:max_pages], 1):
        p_path = pages_dir / f"page_{p_idx:03d}.jpg"
        w, h = save_jpeg_atomic(im, p_path, quality=92)
        page_paths.append(str(p_path))
        page_dims.append([w, h])
        page_urls.append(url)

    return page_paths, page_dims, page_urls


def collect_iiif_music_pages(
    session: requests.Session,
    canvases: Sequence[Dict[str, Any]],
    pages_dir: Path,
    max_pages: int = 8,
    max_scan_inspect: int = 24,
    max_dim: int = 1800,
    clip_threshold: float = CLIP_MUSIC_THRESHOLD,
) -> Tuple[List[str], List[List[int]], List[str]]:
    """Collect up to `max_pages` CLIP-verified music pages from IIIF canvases (parallel fetch)."""
    music_cands = [c for c in canvases if not is_non_music_canvas(c.get("label", ""))]
    if not music_cands:
        music_cands = list(canvases)
    cands = music_cands[:max_scan_inspect]

    def _process_canvas(idx_c: Tuple[int, Dict[str, Any]]) -> Optional[Tuple[int, Image.Image, str]]:
        idx, c = idx_c
        s_local = requests.Session()
        s_local.headers["User-Agent"] = USER_AGENT
        try:
            for u in candidate_iiif_image_urls(c, max_dim=max_dim):
                r = http_get(s_local, u, headers={"User-Agent": USER_AGENT}, timeout=30.0, retries=2)
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
                ok, _stats = _clip_check_safe(im, threshold=clip_threshold)
                if not ok:
                    return None
                return (idx, resize_pil_max_dim(im, max_dim=max_dim), u)
            return None
        finally:
            s_local.close()

    collected: List[Tuple[int, Image.Image, str]] = []
    for start in range(0, len(cands), 8):
        if len(collected) >= max_pages:
            break
        batch = list(enumerate(cands[start : start + 8], start))
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            for res in ex.map(_process_canvas, batch):
                if res is not None:
                    collected.append(res)

    collected.sort(key=lambda x: x[0])
    page_paths: List[str] = []
    page_dims: List[List[int]] = []
    page_urls: List[str] = []
    for p_idx, (_orig_i, im, url) in enumerate(collected[:max_pages], 1):
        p_path = pages_dir / f"page_{p_idx:03d}.jpg"
        w, h = save_jpeg_atomic(im, p_path, quality=92)
        page_paths.append(str(p_path))
        page_dims.append([w, h])
        page_urls.append(url)

    return page_paths, page_dims, page_urls


def resolve_rism_iiif_manifest_url(
    session: requests.Session,
    rism_source_url: str,
    lib_siglum: str,
) -> Optional[str]:
    r = http_get(session, rism_source_url, headers={"Accept": "application/ld+json", "User-Agent": USER_AGENT}, timeout=20.0)
    if r is None:
        return None
    txt = r.text
    r.close()
    urls = sorted(set(re.findall(r'https?://[^\s"<>\\]+', txt)))

    for u in urls:
        if "iiif" in u.lower() and "manifest" in u.lower() and is_allowed_url(u):
            return u

    if lib_siglum == "D-B":
        m_ppn = re.search(r"PPN([0-9]+X?)", txt)
        if m_ppn:
            return f"https://content.staatsbibliothek-berlin.de/dc/PPN{m_ppn.group(1)}/manifest"
        for u in urls:
            if "resolver.staatsbibliothek-berlin.de" in u or "digital.staatsbibliothek-berlin.de" in u:
                rr = http_get(session, u, timeout=20.0, retries=2)
                if rr is not None:
                    final_url = rr.url
                    rr.close()
                    m2 = re.search(r"PPN([0-9]+X?)", final_url)
                    if m2:
                        return f"https://content.staatsbibliothek-berlin.de/dc/PPN{m2.group(1)}/manifest"

    if lib_siglum == "D-Dl":
        for u in urls:
            m_slub = re.search(r"digital\.slub-dresden\.de/id([0-9X]+)", u)
            if m_slub:
                return f"https://iiif.slub-dresden.de/iiif/2/{m_slub.group(1)}/manifest.json"

    return None


def convert_musicxml_to_bekern(
    xml_path: Path,
    bekern_dir: Path,
    slug: str,
    assembly_bin: Path,
) -> Tuple[bool, Optional[Path], int, str]:
    """Convert MusicXML -> post-I1 canonical bekern + verify bekern-decode rc=0."""
    bekern_dir.mkdir(parents=True, exist_ok=True)
    final_bk = bekern_dir / f"{slug}-bekern.txt"

    with tempfile.TemporaryDirectory(prefix="bc_bk_") as td:
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
        if not recanon_i1.is_stream_canonical_i1(bk_text):
            new_toks, changed = recanon_i1.canonicalize_bekern_tokens(bk_text.split())
            if changed:
                bk_text = " ".join(new_toks)
                tmp_bk.write_text(bk_text + "\n", encoding="utf-8")
        if not recanon_i1.is_stream_canonical_i1(bk_text):
            return False, None, len(bk_text.split()), "non_canonical_i1"

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


def convert_krn_to_musicxml(krn_path: Path, out_xml_path: Path) -> Tuple[bool, str]:
    """Convert a CC-BY 4.0 Humdrum **kern file to MusicXML via music21 (no pickle cache)."""
    import music21

    out_xml_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_xml_path.with_suffix(".musicxml.tmp")
    try:
        sc = music21.converter.parse(str(krn_path), storePickle=False)
        sc.write("musicxml", fp=str(tmp))
        os.replace(tmp, out_xml_path)
        return True, ""
    except Exception as e:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass
        return False, f"music21_err:{type(e).__name__}:{str(e)[:140]}"


# ---------------------------------------------------------------------------
# Corpus 1: Bach Chorales Figured Bass v2.0 (CC-BY 4.0, Zenodo 5084914)
# ---------------------------------------------------------------------------


@dataclass
class BachChorale:
    bwv_main: int
    bwv_mvt: Optional[int]
    bwv_label: str  # e.g. "BWV 10.07a"
    xml_path: Path


def ensure_bach_chorales_downloaded(session: requests.Session, bach_dir: Path) -> Path:
    """Verify CC-BY 4.0 on Zenodo 5084914 and download/extract BCFB v2.0 if needed."""
    api_url = f"https://zenodo.org/api/records/{ZENODO_BACH_RECORD_ID}"
    r = http_get(session, api_url, headers={"User-Agent": USER_AGENT}, timeout=25.0)
    if r is None:
        raise RuntimeError(f"Failed to query Zenodo API {api_url}")
    rec = r.json()
    r.close()
    lic_id = str((rec.get("metadata") or {}).get("license", {}).get("id", "")).lower()
    if lic_id != "cc-by-4.0":
        raise RuntimeError(f"Zenodo {ZENODO_BACH_RECORD_ID} license is {lic_id!r}, expected 'cc-by-4.0'")

    existing_master = list(bach_dir.rglob("FB_source/musicXML_master/*.musicxml")) if bach_dir.is_dir() else []
    if len(existing_master) >= 130:
        return existing_master[0].parent

    bach_dir.mkdir(parents=True, exist_ok=True)
    files = rec.get("files") or []
    zip_entry = next((f for f in files if str(f.get("key", "")).endswith(".zip")), None)
    if not zip_entry:
        raise RuntimeError(f"No zip file found in Zenodo {ZENODO_BACH_RECORD_ID}")
    dl_url = str((zip_entry.get("links") or {}).get("self", ""))
    zp = bach_dir / "Bach_chorale_FB-v2.0.zip"
    if not zp.is_file() or zp.stat().st_size < 100_000:
        resp = http_get(session, dl_url, headers={"User-Agent": USER_AGENT}, timeout=90.0)
        if resp is None:
            raise RuntimeError(f"Failed to download {dl_url}")
        tmp = zp.with_suffix(".zip.tmp")
        tmp.write_bytes(resp.content)
        resp.close()
        os.replace(tmp, zp)

    with zipfile.ZipFile(zp, "r") as zf:
        zf.extractall(bach_dir)

    master_files = list(bach_dir.rglob("FB_source/musicXML_master/*.musicxml"))
    if not master_files:
        raise RuntimeError(f"No FB_source/musicXML_master/*.musicxml found after extracting {zp}")
    return master_files[0].parent


def index_bach_chorales(master_dir: Path) -> Tuple[Dict[int, List[BachChorale]], int]:
    """Index BCFB MusicXML files by parent BWV number, strictly excluding BWV 66 (suite31)."""
    by_bwv: Dict[int, List[BachChorale]] = {}
    excluded = 0
    seen_key: Set[Tuple[int, Optional[int]]] = set()

    for p in sorted(master_dir.glob("*.musicxml")):
        m = re.match(r"BWV_(\d+)(?:\.(\d+)([a-z])?)?", p.stem, re.I)
        if not m:
            continue
        bwv_main = int(m.group(1))
        bwv_mvt = int(m.group(2)) if m.group(2) else None
        # Strict eval exclusion: bach-bwv66.6
        if bwv_main == 66:
            excluded += 1
            continue
        if (bwv_main, bwv_mvt) in seen_key:
            continue
        seen_key.add((bwv_main, bwv_mvt))
        lbl = f"BWV {bwv_main}.{bwv_mvt:02d}" if bwv_mvt is not None else f"BWV {bwv_main}"
        by_bwv.setdefault(bwv_main, []).append(
            BachChorale(bwv_main=bwv_main, bwv_mvt=bwv_mvt, bwv_label=lbl, xml_path=p)
        )
    return by_bwv, excluded


def rism_search_sources(
    session: requests.Session,
    query: str,
    lib_siglum: str,
    max_pages: int = 10,
    rows: int = 100,
    hide_composite: bool = True,
) -> List[Dict[str, Any]]:
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
        ]
        if hide_composite:
            params.append(("fq", "hide-composite-volumes:true"))
        r = http_get(
            session,
            "https://rism.online/search",
            params=params,
            headers={"Accept": "application/ld+json", "User-Agent": USER_AGENT},
            timeout=25.0,
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
        if len(items) >= total or len(batch) < rows:
            break
    return items


def match_rism_to_bach_chorale(
    rism_item: Dict[str, Any],
    by_bwv: Dict[int, List[BachChorale]],
) -> Optional[Tuple[BachChorale, float, str]]:
    summary = rism_item.get("summary") or {}
    comp_vals = ((summary.get("sourceComposer") or {}).get("value") or {}).get("none") or []
    comp_norm = normalize_text(" ".join(comp_vals))
    if "bach" not in comp_norm or "johann sebastian" not in comp_norm:
        return None

    label_en = ((rism_item.get("label") or {}).get("en") or [""])[0]
    if not label_en:
        return None
    low_lbl = label_en.lower()
    if any(w in low_lbl for w in ("(sketches)", "(fragment", "bwv anh")):
        return None

    # Extract BWV N or BWV N/M
    best: Optional[Tuple[BachChorale, float, str]] = None
    for m in re.finditer(r"\bBWV\s+(\d+)(?:/(\d+))?\b", label_en, re.I):
        bwv_main = int(m.group(1))
        bwv_mvt = int(m.group(2)) if m.group(2) else None
        if bwv_main == 66:
            return None
        chorales = by_bwv.get(bwv_main)
        if not chorales:
            continue
        if bwv_mvt is not None:
            exact = [c for c in chorales if c.bwv_mvt == bwv_mvt]
            if exact:
                cand = (exact[0], 1.5, f"exact_bwv_mvt:BWV {bwv_main}/{bwv_mvt}")
                if best is None or cand[1] > best[1]:
                    best = cand
        else:
            # Parent cantata/chorale match
            cand = (chorales[0], 1.0, f"bwv_parent:BWV {bwv_main}")
            if best is None or cand[1] > best[1]:
                best = cand
    return best


# ---------------------------------------------------------------------------
# Corpus 2: NIFC Chopin First Editions (CC-BY 4.0)
# ---------------------------------------------------------------------------


@dataclass
class ChopinWork:
    catalog_key: str  # e.g. "op:57", "op:10/5", "op:28"
    ops_display: str  # e.g. "Op. 57"
    title: str
    genre: str
    krn_path: Path


def ensure_nifc_chopin_cloned(chopin_dir: Path) -> Path:
    """Shallow-clone NIFC Chopin First Editions repo and strictly verify CC-BY 4.0 LICENSE.txt."""
    if not (chopin_dir / ".git").is_dir():
        chopin_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", CHOPIN_REPO_URL, str(chopin_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
    lic_path = chopin_dir / "LICENSE.txt"
    if not lic_path.is_file():
        raise RuntimeError(f"Missing LICENSE.txt in {chopin_dir}")
    lic_txt = lic_path.read_text(encoding="utf-8", errors="ignore")
    if "creativecommons.org/licenses/by/4.0" not in lic_txt.lower():
        raise RuntimeError(f"NIFC Chopin LICENSE.txt does not declare CC-BY 4.0")
    return chopin_dir / "kern"


def index_nifc_chopin(kern_dir: Path) -> Tuple[Dict[str, ChopinWork], int]:
    """Index & deduplicate NIFC Chopin .krn files by canonical Op. / Op. No. key.

    Strictly excludes Chopin Op. 6 (contains `chopin-mazurka06-2` in suite31).
    """
    works: Dict[str, ChopinWork] = {}
    excluded = 0

    for p in sorted(kern_dir.glob("*.krn")):
        hdr: Dict[str, str] = {}
        try:
            lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()[:35]
        except Exception:
            continue
        for line in lines:
            if line.startswith("!!!") and ":" in line:
                k, v = line[3:].split(":", 1)
                hdr[k.strip()] = v.strip()

        ops_raw = hdr.get("OPS", "")
        onm_raw = hdr.get("ONM", "")
        otl = hdr.get("OTL", "") or p.stem
        agn = hdr.get("AGN", "")

        # Extract opus number from header or filename prefix (e.g. 006-1-KI-002.krn -> op 6, no 2)
        op_num: Optional[int] = None
        no_num: Optional[int] = None

        m_op = re.search(r"op\.?\s*(\d+)", ops_raw, re.I)
        if m_op:
            op_num = int(m_op.group(1))
        else:
            m_fn = re.match(r"^0*(\d{1,2})-", p.name)
            if m_fn:
                op_num = int(m_fn.group(1))

        m_no = re.search(r"(?:no\.?|nr\.?)\s*(\d+)", f"{ops_raw} {onm_raw} {otl}", re.I)
        if m_no:
            no_num = int(m_no.group(1))
        else:
            m_fn_no = re.search(r"-0*(\d{1,2})\.krn$", p.name, re.I)
            if m_fn_no:
                no_num = int(m_fn_no.group(1))

        if op_num is None:
            continue

        # Strict eval exclusion: chopin-mazurka06-2 (exclude entire Op. 6 to prevent any page overlap)
        if op_num == 6:
            excluded += 1
            continue

        # Skip non-solo-piano companion parts (violoncello/violino) if piano part exists
        if any(s in p.name.lower() for s in ("-violoncello", "-violino")):
            continue

        if no_num is not None:
            key_piece = f"op:{op_num}/{no_num}"
            if key_piece not in works:
                works[key_piece] = ChopinWork(
                    catalog_key=key_piece,
                    ops_display=f"Op. {op_num} No. {no_num}",
                    title=otl,
                    genre=agn,
                    krn_path=p,
                )

        key_op = f"op:{op_num}"
        if key_op not in works:
            works[key_op] = ChopinWork(
                catalog_key=key_op,
                ops_display=f"Op. {op_num}",
                title=otl,
                genre=agn,
                krn_path=p,
            )

    return works, excluded


def extract_chopin_catalog_keys(text: str) -> List[str]:
    """Extract candidate Chopin catalog keys (e.g. 'op:10/5', 'op:57') from RISM or Polona metadata."""
    keys: List[str] = []
    norm = normalize_text(text)

    # 1. Explicit Op. N No./nr M
    for m in re.finditer(r"\bop\.?\s*(\d+)\s*(?:no|nr|n)\.?\s*(\d+)\b", norm):
        k = f"op:{int(m.group(1))}/{int(m.group(2))}"
        if k not in keys:
            keys.append(k)

    # 2. Explicit Op. N
    for m in re.finditer(r"\bop\.?\s*(\d+)\b", norm):
        k = f"op:{int(m.group(1))}"
        if k not in keys:
            keys.append(k)

    # 3. RISM KobC N (Kobylańska catalog uses opus number directly for Op. 1..74, e.g. KobC 58 -> Op. 58)
    for m in re.finditer(r"\bkobc\s+(\d+)\b", norm):
        n = int(m.group(1))
        if 1 <= n <= 74:
            k = f"op:{n}"
            if k not in keys:
                keys.append(k)

    # 4. Known title / dedication hints
    for hint, k in CHOPIN_TITLE_HINTS:
        if hint in norm and k not in keys:
            keys.append(k)

    return keys


def search_polona_chopin_manuscripts(session: requests.Session, max_pages: int = 4) -> List[Dict[str, Any]]:
    """Search Polona2 API for Chopin handwritten music manuscripts."""
    out: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()
    for page_idx in range(max_pages):
        r = http_get(
            session,
            "https://polona2.pl/api/entities/",
            params={
                "query": "Chopin",
                "filters": "public:1",
                "size": "100",
                "from": str(page_idx * 100),
            },
            headers={"User-Agent": USER_AGENT},
            timeout=25.0,
        )
        if r is None:
            break
        try:
            d = r.json()
        except Exception:
            break
        finally:
            r.close()
        hits = d.get("hits") or []
        if not hits:
            break
        for h in hits:
            eid = str(h.get("id") or "")
            if not eid or eid in seen_ids:
                continue
            cats = [str(c).lower() for c in (h.get("categories") or [])]
            if "manuscripts" not in cats:
                continue
            creator = normalize_text(str(h.get("creator") or ""))
            if "chopin" not in creator:
                continue
            title = str(h.get("title") or "")
            alt = str(h.get("alternative") or "")
            tl = normalize_text(f"{title} {alt}")
            # Filter out letters, greetings, calling cards, poetry
            if any(w in tl for w in ("list ", "listy", "powinszowanie", "kartka", "bilet", "sztambuch", "poetyck")):
                continue
            seen_ids.add(eid)
            out.append(h)
    return out


# ---------------------------------------------------------------------------
# Manifest, Verification & Reporting
# ---------------------------------------------------------------------------


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


def verify_zero_eval_overlap(manifest_path: Path) -> Tuple[int, int]:
    """Assert 0 overlap with suite31_hw_manifest.json and suite31_manifest.json."""
    eval_slugs: Set[str] = set()
    for p in (SUITE31_MANIFEST, SUITE31_HW_MANIFEST, GATE20_MANIFEST):
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            for item in data.get("pieces", []) + data.get("pages", []):
                for k in ("slug", "work_id", "id", "score"):
                    v = str(item.get(k, "")).strip().lower()
                    if v:
                        eval_slugs.add(v)
                        if v.startswith("hw-"):
                            eval_slugs.add(v[3:])

    extra_eval = load_eval_slugs()
    total = 0
    overlaps = 0
    if not manifest_path.is_file():
        return 0, 0

    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        total += 1
        assert rec.get("license_verified") is True, f"Unverified license in {rec.get('slug')}"
        slug = str(rec.get("slug", "")).lower()
        cat = str(rec.get("catalog_key", "")).lower()
        title = str(rec.get("title", "")).lower()
        composer = str(rec.get("composer", "")).lower()

        if slug in eval_slugs:
            overlaps += 1
        if "bwv 66" in title or "bwv66" in slug or cat == "bwv:66":
            overlaps += 1
        if "chopin" in composer and (cat == "op:6" or cat.startswith("op:6/") or "mazurka06-2" in slug):
            overlaps += 1
        excl, _ = is_eval_excluded(f"{composer} {title}", slug, extra_slugs=extra_eval)
        if excl:
            overlaps += 1

    assert overlaps == 0, f"Found {overlaps} eval overlaps in {manifest_path}!"
    return total, overlaps


def write_usage_md(out_dir: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# Bach Chorales (CC-BY 4.0) + NIFC Chopin (CC-BY 4.0) × PDM 1.0 Autograph Manuscripts",
        "",
        "## License & Provenance (100% Commercial-Safe)",
        "- **Manuscript Images**: 100% programmatically verified commercial-safe Public Domain scans:",
        "  - **SBB Berlin (`D-B`, `staatsbibliothek-berlin.de`)** — explicit `PDM 1.0`",
        "  - **SLUB Dresden (`D-Dl`, `slub-dresden.de`)** — explicit `PDM 1.0`",
        "  - **Polona / National Library of Poland (`PL-Wn`, `polona2.pl`)** — explicit Polish `Domena Publiczna` with commercial-use permission (`PDM 1.0`)",
        "  - Zero Gallica, zero NC, zero InC (hard domain allowlist + blocklist + per-item manifest/entity rights check; raw manifests archived in `manifests_iiif/`).",
        "- **Symbolic Ground-Truth Scores (`CC-BY 4.0`)**:",
        "  1. **Bach Chorales Figured Bass (BCFB) v2.0** — Zenodo record `5084914` (`CC-BY 4.0`, Yaolong Ju et al.).",
        "  2. **First Editions of Fryderyk Chopin's Music** — Copyright 2017-2021 The Fryderyk Chopin Institute (`https://nifc.pl`), Website `https://chopinscores.org`, Digital scores `https://github.com/pl-wnifc/humdrum-chopin-first-editions`, License `https://creativecommons.org/licenses/by/4.0`.",
        "- **Canonical `bekern`**: Every matched work is exported to `xml/{slug}.musicxml` and converted to post-`I1` canonical `bekern` (`bekern/{slug}-bekern.txt`) via `assembly-serialize.exe bekern` with 100% `bekern-decode` rc=0 verification.",
        "- **Eval Isolation**: `eval_overlap_count = 0` against `suite31_hw_manifest.json` and `suite31_manifest.json` (explicitly excludes `bach-bwv66.6` and `chopin-mazurka06-2` / Chopin Op. 6).",
        "",
        "## Dataset Summary",
        f"- **Total works harvested**: {summary['works_matched']} ({summary['portrait_works']} portrait, {summary['landscape_works']} landscape)",
        f"- **Total CLIP-verified music pages**: {summary['total_pages']} ({summary['portrait_pages']} portrait, {summary['landscape_pages']} landscape; portrait ratio={summary['portrait_page_ratio']*100:.1f}%)",
        f"- **By corpus**: `{json.dumps(summary['by_corpus'])}`",
        f"- **By library**: `{json.dumps(summary['by_library'])}`",
        f"- **By rights**: `{json.dumps(summary['by_rights'])}`",
        f"- **bekern-decode pass rate**: {summary['bekern_decode_passed']}/{summary['works_matched']} ({summary['bekern_decode_pass_rate']*100:.1f}%)",
        f"- **Eval overlap**: `{summary['eval_overlap_count']}`",
        "",
        "## Directory Layout",
        "- `pages/{slug}/page_001.jpg` ... — CLIP-verified handwritten music pages (`<= 1800 px`, RGB JPEG Q=92)",
        "- `xml/{slug}.musicxml` — `CC-BY 4.0` MusicXML ground truth",
        "- `bekern/{slug}-bekern.txt` — post-`I1` canonical `bekern` tokens (`bekern-decode` rc=0)",
        "- `manifests_iiif/{slug}.json` — archived IIIF manifest / Polona2 entity metadata with rights proof",
        "- `manifest.jsonl` — per-work metadata, provenance, rights evidence, page paths & dimensions",
        "- `summary.json` — aggregate counts and verification metrics",
        "",
        "## Regeneration / Resume",
        "```powershell",
        '$env:CUDA_VISIBLE_DEVICES=""; .\\.venv\\Scripts\\python.exe tmp-train/mvd/harvest_bach_chopin_hw.py --out D:\\tmp\\pdm-bach-chopin-hw --max-pages-per-work 8 --resume',
        "```",
        "",
    ]
    tmp = out_dir / "USAGE.md.tmp"
    tmp.write_text("\n".join(lines), encoding="utf-8")
    os.replace(tmp, out_dir / "USAGE.md")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Harvest Bach Chorales (CC-BY 4.0) + NIFC Chopin (CC-BY 4.0) x PDM 1.0 handwritten manuscripts."
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--bach-dir", type=Path, default=DEFAULT_BACH_DIR)
    ap.add_argument("--chopin-dir", type=Path, default=DEFAULT_CHOPIN_DIR)
    ap.add_argument("--assembly-bin", type=Path, default=DEFAULT_ASSEMBLY_BIN)
    ap.add_argument("--max-bach-works", type=int, default=40)
    ap.add_argument("--max-chopin-works", type=int, default=40)
    ap.add_argument("--max-pages-per-work", type=int, default=8)
    ap.add_argument("--max-dim", type=int, default=1800)
    ap.add_argument("--clip-threshold", type=float, default=CLIP_MUSIC_THRESHOLD)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.out.resolve() == Path(r"D:\tmp\rism-pdm-hw").resolve():
        print("ERROR: Refusing to touch D:\\tmp\\rism-pdm-hw", file=sys.stderr)
        return 1
    if not args.assembly_bin.is_file():
        print(f"ERROR: assembly-serialize binary not found at {args.assembly_bin}", file=sys.stderr)
        return 1

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"
    existing = load_existing_manifest(manifest_path) if args.resume else {}
    if not args.resume and manifest_path.is_file() and not args.dry_run:
        manifest_path.unlink()

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    eval_slugs = load_eval_slugs()

    # 1. Prepare & verify CC-BY 4.0 symbolic corpora
    print("[1/5] Preparing & verifying CC-BY 4.0 symbolic corpora ...", flush=True)
    bach_master_dir = ensure_bach_chorales_downloaded(session, args.bach_dir)
    bach_by_bwv, bach_excl = index_bach_chorales(bach_master_dir)
    n_bach_files = sum(len(v) for v in bach_by_bwv.values())
    print(f"      Bach BCFB v2.0 (CC-BY 4.0): {n_bach_files} chorales across {len(bach_by_bwv)} unique BWVs (excluded={bach_excl})", flush=True)

    chopin_kern_dir = ensure_nifc_chopin_cloned(args.chopin_dir)
    chopin_works, chopin_excl = index_nifc_chopin(chopin_kern_dir)
    print(f"      NIFC Chopin (CC-BY 4.0): {len(chopin_works)} catalog keys indexed (Op.6 excluded={chopin_excl})", flush=True)

    if not args.dry_run:
        t0 = time.time()
        get_clip_classifier()
        print(f"[clip] CPU CLIP classifier ready in {time.time() - t0:.1f}s", flush=True)

    downloaded_records: Dict[str, Dict[str, Any]] = dict(existing)
    seen_manifests: Set[str] = {
        str(r.get("iiif_manifest") or r.get("polona_api_url") or "")
        for r in downloaded_records.values()
        if r.get("iiif_manifest") or r.get("polona_api_url")
    }

    # -----------------------------------------------------------------------
    # 2. Harvest Bach Chorales x SBB Berlin (D-B) + SLUB Dresden (D-Dl)
    # -----------------------------------------------------------------------
    print("[2/5] Searching RISM Online for Bach manuscripts in D-B (SBB Berlin) + D-Dl (SLUB Dresden) ...", flush=True)
    bach_cands: List[Dict[str, Any]] = []
    for lib, max_p in (("D-B", 25), ("D-Dl", 3)):
        rism_items = rism_search_sources(session, "Johann Sebastian Bach", lib, max_pages=max_p, rows=100, hide_composite=True)
        matched_here = 0
        for it in rism_items:
            res = match_rism_to_bach_chorale(it, bach_by_bwv)
            if not res:
                continue
            chorale, score, reason = res
            rism_url = str(it.get("id") or "")
            label_en = ((it.get("label") or {}).get("en") or [""])[0]
            mat_types = [
                ((m.get("typeLabel") or {}).get("en") or [""])[0]
                for m in ((it.get("flags") or {}).get("materialTypes") or [])
            ]
            is_auto = any("autograph" in m.lower() for m in mat_types)
            # Prefer standalone manuscripts over composite Am.B 46/II volume entries
            is_composite = "am.b 46" in label_en.lower()
            eff_score = score + (0.4 if is_auto else 0.0) - (0.35 if is_composite else 0.0)
            bach_cands.append(
                {
                    "chorale": chorale,
                    "score": eff_score,
                    "reason": reason,
                    "lib_siglum": lib,
                    "library": "SBB Berlin" if lib == "D-B" else "SLUB Dresden",
                    "rism_url": rism_url,
                    "rism_label": label_en,
                    "material_type": ", ".join(m for m in mat_types if m) or "Manuscript",
                }
            )
            matched_here += 1
        print(f"      Bach {lib}: scanned {len(rism_items)} RISM items -> {matched_here} BWV matches", flush=True)

    bach_cands.sort(key=lambda c: c["score"], reverse=True)
    bach_saved = sum(1 for r in downloaded_records.values() if r.get("corpus") == "bach_chorales_fb")
    seen_bach_bwv: Set[str] = {
        str(r.get("catalog_key")) for r in downloaded_records.values() if r.get("corpus") == "bach_chorales_fb"
    }

    print(f"[3/5] Verifying PDM 1.0 & harvesting Bach Chorales (target up to {args.max_bach_works}) ...", flush=True)
    for cand in bach_cands:
        if bach_saved >= args.max_bach_works:
            break
        chorale: BachChorale = cand["chorale"]
        cat_key = f"bwv:{chorale.bwv_main}"
        if cat_key in seen_bach_bwv:
            continue

        slug = slugify(f"bach-{chorale.bwv_label}-{cand['lib_siglum'].lower()}")
        if args.resume and slug in downloaded_records:
            seen_bach_bwv.add(cat_key)
            continue

        mf_url = resolve_rism_iiif_manifest_url(session, cand["rism_url"], cand["lib_siglum"])
        if not mf_url or not is_allowed_url(mf_url) or mf_url in seen_manifests:
            continue

        rm = http_get(session, mf_url, headers={"User-Agent": USER_AGENT}, timeout=25.0)
        if rm is None:
            continue
        try:
            mf_json = rm.json()
        except Exception:
            continue
        finally:
            rm.close()

        ok_rights, rights_str, rights_raw = verify_iiif_pdm10(mf_json, mf_url)
        if not ok_rights:
            continue

        canvases = extract_iiif_canvases(mf_json)
        if not canvases:
            continue

        if args.dry_run:
            print(f"      [DRY BACH] {slug:35s} | {cand['library']:12s} | {rights_str} | canvases={len(canvases)} | {cand['rism_label'][:60]}", flush=True)
            seen_manifests.add(mf_url)
            seen_bach_bwv.add(cat_key)
            bach_saved += 1
            continue

        # Convert MusicXML -> bekern first so we only download images when bekern-decode rc=0
        xml_out = out_dir / "xml" / f"{slug}.musicxml"
        xml_out.parent.mkdir(parents=True, exist_ok=True)
        tmp_xml = xml_out.with_suffix(".musicxml.tmp")
        tmp_xml.write_bytes(chorale.xml_path.read_bytes())
        os.replace(tmp_xml, xml_out)

        ok_bk, bk_path, bk_toks, bk_err = convert_musicxml_to_bekern(xml_out, out_dir / "bekern", slug, args.assembly_bin)
        if not ok_bk or bk_path is None:
            print(f"      [SKIP BEKERN] {slug}: {bk_err}", flush=True)
            continue

        pages_dir = out_dir / "pages" / slug
        page_paths, page_dims, page_urls = collect_iiif_music_pages(
            session,
            canvases,
            pages_dir,
            max_pages=args.max_pages_per_work,
            max_scan_inspect=20,
            max_dim=args.max_dim,
            clip_threshold=args.clip_threshold,
        )
        if not page_paths:
            continue

        mf_out = out_dir / "manifests_iiif" / f"{slug}.json"
        mf_out.parent.mkdir(parents=True, exist_ok=True)
        mf_out.write_text(json.dumps(mf_json, ensure_ascii=False, indent=2), encoding="utf-8")

        n_port = sum(1 for dw, dh in page_dims if dh > dw)
        n_land = sum(1 for dw, dh in page_dims if dw >= dh)
        orient = "portrait" if n_port >= n_land else "landscape"

        rec = {
            "slug": slug,
            "corpus": "bach_chorales_fb",
            "composer": "Bach, Johann Sebastian",
            "title": chorale.bwv_label,
            "catalog_key": cat_key,
            "library": cand["library"],
            "library_siglum": cand["lib_siglum"],
            "iiif_manifest": mf_url,
            "rights": rights_str,
            "license_verified": True,
            "license_raw": rights_raw,
            "xml_license": "CC-BY-4.0 (Zenodo 5084914 Bach Chorales Figured Bass v2.0)",
            "xml_path": str(xml_out),
            "bekern_path": str(bk_path),
            "bekern_tokens": bk_toks,
            "bekern_decode_rc": 0,
            "n_pages": len(page_paths),
            "orientation": orient,
            "n_portrait_pages": n_port,
            "n_landscape_pages": n_land,
            "page_paths": page_paths,
            "page_dims": page_dims,
            "page_urls": page_urls,
            "rism_url": cand["rism_url"],
            "rism_label": cand["rism_label"],
            "material_type": cand["material_type"],
            "match_reason": cand["reason"],
            "eval_overlap": False,
        }
        downloaded_records[slug] = rec
        seen_manifests.add(mf_url)
        seen_bach_bwv.add(cat_key)
        bach_saved += 1
        with manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(
            f"      [SAVED BACH {bach_saved:2d}] {slug:32s} | {cand['library']:12s} | {len(page_paths)}p ({n_port}P/{n_land}L) | bk={bk_toks} tok | {rights_str}",
            flush=True,
        )

    # -----------------------------------------------------------------------
    # 3. Harvest NIFC Chopin x Polona (PL-Wn) + SBB Berlin (D-B)
    # -----------------------------------------------------------------------
    print("[4/5] Searching Polona2 API (PL-Wn) + RISM (PL-Wn, D-B) for Chopin manuscripts ...", flush=True)
    chopin_polona_ids: Dict[str, Dict[str, Any]] = {}

    # 3a. Direct Polona2 search for Chopin manuscripts
    pol_hits = search_polona_chopin_manuscripts(session, max_pages=4)
    print(f"      Polona2 search returned {len(pol_hits)} Chopin manuscript candidates", flush=True)
    for h in pol_hits:
        eid = str(h.get("id") or "")
        if eid:
            chopin_polona_ids[eid] = {
                "source": "polona2_search",
                "title": f"{h.get('title') or ''} | {h.get('alternative') or ''}".strip(" |"),
                "rism_url": "",
                "rism_label": "",
                "material_type": "Manuscript (Polona)",
            }

    # 3b. RISM PL-Wn Chopin manuscripts -> resolve Polona ID
    rism_pl = rism_search_sources(session, "Chopin", "PL-Wn", max_pages=3, rows=100, hide_composite=False)
    print(f"      RISM PL-Wn returned {len(rism_pl)} Chopin manuscript records", flush=True)
    for it in rism_pl:
        rism_url = str(it.get("id") or "")
        lbl = ((it.get("label") or {}).get("en") or [""])[0]
        r_src = http_get(session, rism_url, headers={"Accept": "application/ld+json", "User-Agent": USER_AGENT}, timeout=20.0)
        if r_src is None:
            continue
        try:
            src_json = r_src.json()
        except Exception:
            continue
        finally:
            r_src.close()
        pid = extract_polona_item_id(src_json)
        if pid and pid not in chopin_polona_ids:
            chopin_polona_ids[pid] = {
                "source": "rism_pl_wn",
                "title": lbl,
                "rism_url": rism_url,
                "rism_label": lbl,
                "material_type": "Autograph/Manuscript (RISM PL-Wn)",
            }

    # 3c. Process Polona Chopin manuscripts
    chopin_saved = sum(1 for r in downloaded_records.values() if r.get("corpus") == "nifc_chopin")
    seen_chopin_keys: Set[str] = {
        str(r.get("catalog_key")) for r in downloaded_records.values() if r.get("corpus") == "nifc_chopin"
    }

    print(f"[5/5] Verifying licenses & harvesting Chopin manuscripts (Polona={len(chopin_polona_ids)} + D-B) ...", flush=True)
    for pid, meta in chopin_polona_ids.items():
        if chopin_saved >= args.max_chopin_works:
            break
        api_url = f"https://polona2.pl/api/entities/{pid}"
        if api_url in seen_manifests:
            continue
        r_ent = http_get(session, api_url, headers={"User-Agent": USER_AGENT}, timeout=20.0)
        if r_ent is None:
            continue
        try:
            entity = r_ent.json()
        except Exception:
            continue
        finally:
            r_ent.close()

        lic_ok, lic_ev = verify_polona_license(entity)
        if not lic_ok:
            continue

        full_text = " ".join(
            [
                meta.get("title", ""),
                meta.get("rism_label", ""),
                str(entity.get("title") or ""),
                str(entity.get("alternative") or ""),
                " ".join(_flatten_strings(entity.get("uniform_title"))),
                " ".join(_flatten_strings(entity.get("notes"))),
            ]
        )
        # Check eval exclusion
        cand_keys = extract_chopin_catalog_keys(full_text)
        if any(k == "op:6" or k.startswith("op:6/") for k in cand_keys):
            continue

        matched_work: Optional[ChopinWork] = None
        for k in cand_keys:
            if k in chopin_works and k not in seen_chopin_keys:
                matched_work = chopin_works[k]
                break
        if matched_work is None:
            continue

        excl, _ = is_eval_excluded(f"Chopin {matched_work.ops_display} {full_text}", pid, extra_slugs=eval_slugs)
        if excl:
            continue

        slug = slugify(f"chopin-{matched_work.ops_display}-pl-wn-{pid[:8]}")
        if args.resume and slug in downloaded_records:
            seen_chopin_keys.add(matched_work.catalog_key)
            continue

        scans = entity.get("scans") or []
        if not scans:
            continue

        if args.dry_run:
            print(f"      [DRY CHOPIN PL-Wn] {slug:35s} | {matched_work.ops_display:12s} | scans={len(scans)} | {lic_ev[:35]}", flush=True)
            seen_manifests.add(api_url)
            seen_chopin_keys.add(matched_work.catalog_key)
            chopin_saved += 1
            continue

        xml_out = out_dir / "xml" / f"{slug}.musicxml"
        ok_xml, xml_err = convert_krn_to_musicxml(matched_work.krn_path, xml_out)
        if not ok_xml:
            print(f"      [SKIP XML] {slug}: {xml_err}", flush=True)
            continue

        ok_bk, bk_path, bk_toks, bk_err = convert_musicxml_to_bekern(xml_out, out_dir / "bekern", slug, args.assembly_bin)
        if not ok_bk or bk_path is None:
            print(f"      [SKIP BEKERN] {slug}: {bk_err}", flush=True)
            continue

        pages_dir = out_dir / "pages" / slug
        page_paths, page_dims, page_urls = collect_polona_music_pages(
            session,
            scans,
            pages_dir,
            max_pages=args.max_pages_per_work,
            max_scan_inspect=24,
            max_dim=args.max_dim,
            clip_threshold=args.clip_threshold,
        )
        if not page_paths:
            continue

        # Archive Polona entity metadata (excluding huge scan arrays if desired, keep rights + title + scan count)
        ent_archive = {
            "id": entity.get("id"),
            "title": entity.get("title"),
            "alternative": entity.get("alternative"),
            "creator": entity.get("creator"),
            "rights": entity.get("rights"),
            "categories": entity.get("categories"),
            "total_scans": len(scans),
            "polona_api_url": api_url,
        }
        mf_out = out_dir / "manifests_iiif" / f"{slug}.json"
        mf_out.parent.mkdir(parents=True, exist_ok=True)
        mf_out.write_text(json.dumps(ent_archive, ensure_ascii=False, indent=2), encoding="utf-8")

        n_port = sum(1 for dw, dh in page_dims if dh > dw)
        n_land = sum(1 for dw, dh in page_dims if dw >= dh)
        orient = "portrait" if n_port >= n_land else "landscape"

        rec = {
            "slug": slug,
            "corpus": "nifc_chopin",
            "composer": "Chopin, Fryderyk",
            "title": f"{matched_work.ops_display} ({matched_work.title})",
            "catalog_key": matched_work.catalog_key,
            "library": "Polona",
            "library_siglum": "PL-Wn",
            "polona_api_url": api_url,
            "polona_item_id": pid,
            "rights": lic_ev,
            "license_verified": True,
            "license_raw": " | ".join(_flatten_strings(entity.get("rights")))[:240],
            "xml_license": "CC-BY-4.0 (NIFC humdrum-chopin-first-editions)",
            "source_krn": str(matched_work.krn_path),
            "xml_path": str(xml_out),
            "bekern_path": str(bk_path),
            "bekern_tokens": bk_toks,
            "bekern_decode_rc": 0,
            "n_pages": len(page_paths),
            "orientation": orient,
            "n_portrait_pages": n_port,
            "n_landscape_pages": n_land,
            "page_paths": page_paths,
            "page_dims": page_dims,
            "page_urls": page_urls,
            "rism_url": meta.get("rism_url", ""),
            "rism_label": meta.get("rism_label", ""),
            "material_type": meta.get("material_type", "Manuscript (Polona)"),
            "eval_overlap": False,
        }
        downloaded_records[slug] = rec
        seen_manifests.add(api_url)
        seen_chopin_keys.add(matched_work.catalog_key)
        chopin_saved += 1
        with manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(
            f"      [SAVED CHOPIN {chopin_saved:2d}] {slug:34s} | Polona       | {len(page_paths)}p ({n_port}P/{n_land}L) | bk={bk_toks} tok",
            flush=True,
        )

    # 3d. Also check SBB Berlin (D-B) Chopin manuscripts via RISM
    rism_db_chopin = rism_search_sources(session, "Chopin", "D-B", max_pages=3, rows=100, hide_composite=False)
    print(f"      RISM D-B returned {len(rism_db_chopin)} Chopin manuscript records", flush=True)
    for it in rism_db_chopin:
        if chopin_saved >= args.max_chopin_works:
            break
        summary = it.get("summary") or {}
        comp_vals = ((summary.get("sourceComposer") or {}).get("value") or {}).get("none") or []
        if "chopin" not in normalize_text(" ".join(comp_vals)):
            continue
        lbl = ((it.get("label") or {}).get("en") or [""])[0]
        if "(arrangement)" in lbl.lower():
            continue
        cand_keys = extract_chopin_catalog_keys(lbl)
        if any(k == "op:6" or k.startswith("op:6/") for k in cand_keys):
            continue
        matched_work = None
        for k in cand_keys:
            if k in chopin_works and k not in seen_chopin_keys:
                matched_work = chopin_works[k]
                break
        if matched_work is None:
            continue

        rism_url = str(it.get("id") or "")
        mf_url = resolve_rism_iiif_manifest_url(session, rism_url, "D-B")
        if not mf_url or not is_allowed_url(mf_url) or mf_url in seen_manifests:
            continue
        rm = http_get(session, mf_url, headers={"User-Agent": USER_AGENT}, timeout=25.0)
        if rm is None:
            continue
        try:
            mf_json = rm.json()
        except Exception:
            continue
        finally:
            rm.close()

        ok_rights, rights_str, rights_raw = verify_iiif_pdm10(mf_json, mf_url)
        if not ok_rights:
            continue
        canvases = extract_iiif_canvases(mf_json)
        if not canvases:
            continue

        slug = slugify(f"chopin-{matched_work.ops_display}-d-b")
        if args.resume and slug in downloaded_records:
            seen_chopin_keys.add(matched_work.catalog_key)
            continue

        if args.dry_run:
            print(f"      [DRY CHOPIN D-B] {slug:35s} | {matched_work.ops_display:12s} | canvases={len(canvases)} | {rights_str}", flush=True)
            seen_manifests.add(mf_url)
            seen_chopin_keys.add(matched_work.catalog_key)
            chopin_saved += 1
            continue

        xml_out = out_dir / "xml" / f"{slug}.musicxml"
        ok_xml, xml_err = convert_krn_to_musicxml(matched_work.krn_path, xml_out)
        if not ok_xml:
            continue
        ok_bk, bk_path, bk_toks, bk_err = convert_musicxml_to_bekern(xml_out, out_dir / "bekern", slug, args.assembly_bin)
        if not ok_bk or bk_path is None:
            continue

        pages_dir = out_dir / "pages" / slug
        page_paths, page_dims, page_urls = collect_iiif_music_pages(
            session,
            canvases,
            pages_dir,
            max_pages=args.max_pages_per_work,
            max_scan_inspect=20,
            max_dim=args.max_dim,
            clip_threshold=args.clip_threshold,
        )
        if not page_paths:
            continue

        mf_out = out_dir / "manifests_iiif" / f"{slug}.json"
        mf_out.parent.mkdir(parents=True, exist_ok=True)
        mf_out.write_text(json.dumps(mf_json, ensure_ascii=False, indent=2), encoding="utf-8")

        n_port = sum(1 for dw, dh in page_dims if dh > dw)
        n_land = sum(1 for dw, dh in page_dims if dw >= dh)
        orient = "portrait" if n_port >= n_land else "landscape"

        rec = {
            "slug": slug,
            "corpus": "nifc_chopin",
            "composer": "Chopin, Fryderyk",
            "title": f"{matched_work.ops_display} ({matched_work.title})",
            "catalog_key": matched_work.catalog_key,
            "library": "SBB Berlin",
            "library_siglum": "D-B",
            "iiif_manifest": mf_url,
            "rights": rights_str,
            "license_verified": True,
            "license_raw": rights_raw,
            "xml_license": "CC-BY-4.0 (NIFC humdrum-chopin-first-editions)",
            "source_krn": str(matched_work.krn_path),
            "xml_path": str(xml_out),
            "bekern_path": str(bk_path),
            "bekern_tokens": bk_toks,
            "bekern_decode_rc": 0,
            "n_pages": len(page_paths),
            "orientation": orient,
            "n_portrait_pages": n_port,
            "n_landscape_pages": n_land,
            "page_paths": page_paths,
            "page_dims": page_dims,
            "page_urls": page_urls,
            "rism_url": rism_url,
            "rism_label": lbl,
            "material_type": "Manuscript (SBB Berlin)",
            "eval_overlap": False,
        }
        downloaded_records[slug] = rec
        seen_manifests.add(mf_url)
        seen_chopin_keys.add(matched_work.catalog_key)
        chopin_saved += 1
        with manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(
            f"      [SAVED CHOPIN {chopin_saved:2d}] {slug:34s} | SBB Berlin   | {len(page_paths)}p ({n_port}P/{n_land}L) | bk={bk_toks} tok",
            flush=True,
        )

    if args.dry_run:
        print(f"\nDRY RUN complete: Bach={bach_saved}, Chopin={chopin_saved}")
        return 0

    # Verify zero eval overlap
    total_ver, overlaps = verify_zero_eval_overlap(manifest_path)

    recs = list(downloaded_records.values())
    total_pages = sum(r["n_pages"] for r in recs)
    total_port_pages = sum(r.get("n_portrait_pages", 0) for r in recs)
    total_land_pages = sum(r.get("n_landscape_pages", 0) for r in recs)
    by_corpus: Dict[str, Dict[str, int]] = {}
    by_lib: Dict[str, int] = {}
    by_rights: Dict[str, int] = {}
    bk_passed = 0
    for r in recs:
        c = r.get("corpus", "unknown")
        by_corpus.setdefault(c, {"works": 0, "pages": 0})
        by_corpus[c]["works"] += 1
        by_corpus[c]["pages"] += r["n_pages"]
        by_lib[r["library"]] = by_lib.get(r["library"], 0) + 1
        by_rights[r["rights"]] = by_rights.get(r["rights"], 0) + 1
        if r.get("bekern_decode_rc") == 0 and Path(r.get("bekern_path", "")).is_file():
            bk_passed += 1

    summary = {
        "works_matched": len(recs),
        "portrait_works": sum(1 for r in recs if r["orientation"] == "portrait"),
        "landscape_works": sum(1 for r in recs if r["orientation"] == "landscape"),
        "total_pages": total_pages,
        "portrait_pages": total_port_pages,
        "landscape_pages": total_land_pages,
        "portrait_page_ratio": round(total_port_pages / total_pages, 4) if total_pages else 0.0,
        "by_corpus": by_corpus,
        "by_library": by_lib,
        "by_rights": by_rights,
        "bekern_decode_passed": bk_passed,
        "bekern_decode_pass_rate": round(bk_passed / len(recs), 4) if recs else 0.0,
        "eval_overlap_count": overlaps,
        "manifest_verified_rows": total_ver,
    }
    tmp_sum = out_dir / "summary.json.tmp"
    tmp_sum.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_sum, out_dir / "summary.json")
    write_usage_md(out_dir, summary)

    print("\n=== HARVEST SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
