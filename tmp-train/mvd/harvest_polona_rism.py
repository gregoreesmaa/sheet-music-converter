#!/usr/bin/env python3
"""Polona (PL-Wn) Commercial-Safe RISM Manuscript Harvester (CPU-only).

Harvests strictly commercial-safe (Polish PDM 1.0 / Domena Publiczna with explicit
commercial-use permission) handwritten music manuscripts from PL-Wn (National
Library of Poland / Polona) paired with RISM Online CC-BY 4.0 PAE/MEI incipits
converted to post-I1 canonical bekern (bekern-decode rc == 0).

Writes into D:\\tmp\\rism-pdm-hw-extra\\ (never touches D:\\tmp\\rism-pdm-hw\\).
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import argparse
import concurrent.futures
import io
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import requests
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import recanon_i1  # noqa: E402
from harvest_rism_pdm import (  # noqa: E402
    DEFAULT_ASSEMBLY_BIN,
    RISM_HEADERS,
    RismHarvestState,
    _flatten_strings,
    convert_incipit_to_bekern,
    fetch_institution_source_urls,
    http_get,
)
from harvest_unlabeled_10k import (  # noqa: E402
    CLIP_MUSIC_THRESHOLD,
    get_clip_classifier,
    is_eval_excluded,
    is_music_page,
    load_eval_slugs,
    resize_and_encode_jpeg,
)
from harvest_pdm_openscore import is_held_out_hw, normalize_text  # noqa: E402

USER_AGENT = "sheet-music-converter-research/1.0 (RISM-PL-Wn-Polona-PDM; CPU-only)"
DEFAULT_OUT_DIR = Path(r"D:\tmp\rism-pdm-hw-extra")
SUITE31_HW_MANIFEST = REPO_ROOT / "data-gen" / "score-corpus" / "suite31" / "suite31_hw_manifest.json"

POLONA_ALLOWED_SUFFIXES = ("polona.pl", "polona2.pl", "zas.bn.org.pl")

POLONA_REJECT_PATTERNS = (
    re.compile(r"chronion\w*\s+prawem\s+autorskim", re.I),
    re.compile(r"in[- ]?copyright", re.I),
    re.compile(r"\bby-nc\b", re.I),
    re.compile(r"non[- ]?commercial", re.I),
    re.compile(r"all rights reserved", re.I),
    re.compile(r"dozwolony\s+użytek", re.I),
)


def is_polona_allowed_url(url: str) -> bool:
    if not url or not url.startswith(("http://", "https://")):
        return False
    host = (urlparse(url.lower()).hostname or "").strip(".")
    return any(host == suf or host.endswith("." + suf) for suf in POLONA_ALLOWED_SUFFIXES)


def verify_polona_license(entity: Dict[str, Any]) -> Tuple[bool, str]:
    """Verify explicit commercial-safe Public Domain license on a Polona entity."""
    rights = _flatten_strings(entity.get("rights"))
    joined = " | ".join(rights)
    if not joined:
        return False, "no_rights_metadata"
    for pat in POLONA_REJECT_PATTERNS:
        m = pat.search(joined)
        if m:
            return False, f"rejected_license:{m.group(0)}"
    low = joined.lower()
    if "domena publiczna" in low and ("celach komercyjnych" in low or "commercial" in low):
        return True, "Domena Publiczna (PDM 1.0 - commercial use permitted)"
    if "public domain" in low or "creativecommons.org/publicdomain/mark/1.0" in low or "cc0" in low:
        return True, joined[:140]
    return False, f"unrecognized_license:{joined[:140]}"


def extract_polona_item_id(source_json: Dict[str, Any]) -> Optional[str]:
    ext_items = source_json.get("externalResources", {}).get("items", [])
    if not isinstance(ext_items, list):
        return None
    for item in ext_items:
        url = str(item.get("url", "")).strip()
        if "polona.pl" not in url.lower():
            continue
        m = re.search(r"polona\.pl/item/(\d+)", url, re.I)
        if m:
            return m.group(1)
    return None


def _fetch_polona_image(session: requests.Session, url: str, timeout: float = 25.0) -> Optional[Image.Image]:
    if not is_polona_allowed_url(url):
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


def find_first_polona_music_scan(
    session: requests.Session,
    scans: List[Dict[str, Any]],
    max_scan: int = 8,
    clip_threshold: float = CLIP_MUSIC_THRESHOLD,
) -> Tuple[Optional[Image.Image], Optional[int], str, Dict[str, Any]]:
    if not scans:
        return None, None, "", {}
    n = min(len(scans), max_scan)
    order = list(range(1, n)) + ([0] if n >= 1 else [])

    for idx in order:
        sc = scans[idx]
        if not isinstance(sc, dict):
            continue
        # Pick largest thumbnail (prefer type=huge ~800px) for CLIP preview
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
            preview_img = _fetch_polona_image(session, u, timeout=20.0)
            if preview_img is not None:
                preview_url = u
                break
        if preview_img is None:
            continue

        ok, stats = is_music_page(preview_img, threshold=clip_threshold)
        if not ok:
            continue

        # Download full-res JPEG if available
        full_url = ""
        for res in sc.get("resources") or []:
            if isinstance(res, dict) and res.get("mime") == "image/jpeg" and res.get("url"):
                full_url = str(res["url"])
                break
        if full_url:
            hires = _fetch_polona_image(session, full_url, timeout=35.0)
            if hires is not None:
                return hires, idx, full_url, stats

        if max(preview_img.size) >= 700:
            return preview_img, idx, preview_url, stats

    return None, None, "", {}


def process_polona_source(
    source_url: str,
    state: RismHarvestState,
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
        src_json = r_src.json()
        r_src.close()

        stype = src_json.get("sourceTypes", {}).get("sourceType", {}).get("type", "")
        if stype and "Manuscript" not in stype:
            return None

        label_str = " ".join(_flatten_strings(src_json.get("label")))
        creator_str = " ".join(_flatten_strings(src_json.get("creator")))
        excl, excl_reason = is_eval_excluded(f"{creator_str} {label_str}", rism_id, extra_slugs=eval_slugs)
        if excl or is_held_out_hw(normalize_text(creator_str), normalize_text(label_str)):
            state.bump("eval_excluded", excl_reason or "held_out_hw")
            return None

        polona_id = extract_polona_item_id(src_json)
        if not polona_id:
            state.bump("no_iiif_url", "no_polona_id")
            return None

        api_url = f"https://polona2.pl/api/entities/{polona_id}"
        r_ent = http_get(session, api_url, headers={"User-Agent": USER_AGENT}, timeout=20.0, retries=2)
        if r_ent is None:
            state.bump("no_iiif_url", "polona_api_failed")
            return None
        try:
            entity = r_ent.json()
        except Exception:
            state.bump("no_iiif_url", "polona_bad_json")
            return None
        finally:
            r_ent.close()

        lic_ok, lic_evidence = verify_polona_license(entity)
        if not lic_ok:
            state.bump("license_rejected", lic_evidence)
            return None
        state.bump("license_verified")

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
                # Enforce post-I1 canonical form
                if not recanon_i1.is_stream_canonical_i1(bk):
                    new_toks, changed = recanon_i1.canonicalize_bekern_tokens(bk.split())
                    if changed:
                        bk = " ".join(new_toks)
                if recanon_i1.is_stream_canonical_i1(bk):
                    bekern_txt = bk
                    inc_meta = imeta
                    break
                last_err = "non_canonical_i1"
            else:
                last_err = err

        if bekern_txt is None:
            with state.lock:
                key = (last_err or "pae_failed").split(":")[0]
                state.error_reasons[key] = state.error_reasons.get(key, 0) + 1
            return None
        state.bump("pae_succeeded")

        scans = entity.get("scans") or []
        hires_img, scan_idx, img_url, clip_stats = find_first_polona_music_scan(
            session, scans, max_scan=8, clip_threshold=clip_threshold
        )
        if hires_img is None or scan_idx is None:
            state.bump("no_music_canvas", "no_music_canvas")
            return None

        jpg_bytes, w, h = resize_and_encode_jpeg(hires_img, max_dim=max_dim, quality=90)
        record_meta = {
            "source_url": source_url,
            "title": label_str[:300],
            "creator": creator_str[:200],
            "polona_api_url": api_url,
            "polona_item_id": polona_id,
            "image_url": img_url,
            "total_canvases": len(scans),
            "license_verified": True,
            "license_evidence": lic_evidence,
            "license_raw": " | ".join(_flatten_strings(entity.get("rights")))[:300],
            "clip_stats": clip_stats,
            "bekern": bekern_txt,
            "bekern_decode_rc": 0,
            **inc_meta,
        }
        saved = state.record_pair(rism_id, "PL-Wn", scan_idx, jpg_bytes, w, h, record_meta)
        return rism_id if saved else None
    finally:
        state.mark_processed(rism_id)
        session.close()


def verify_zero_eval_overlap(manifest_path: Path) -> Tuple[int, int]:
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
    if not manifest_path.is_file():
        return 0, 0
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        total += 1
        assert rec.get("license_verified") is True, f"Unverified license in {rec.get('rism_id')}"
        rid = str(rec.get("rism_id", "")).lower()
        if rid in eval_slugs or is_held_out_hw(normalize_text(rec.get("creator", "")), normalize_text(rec.get("title", ""))):
            overlaps += 1
    assert overlaps == 0, f"Found {overlaps} eval overlaps!"
    return total, overlaps


def main() -> int:
    ap = argparse.ArgumentParser(description="Harvest PL-Wn (Polona) PDM 1.0 RISM manuscripts (CPU-only).")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--limit", type=int, default=600)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-dim", type=int, default=1800)
    ap.add_argument("--clip-threshold", type=float, default=CLIP_MUSIC_THRESHOLD)
    ap.add_argument("--assembly-bin", type=Path, default=DEFAULT_ASSEMBLY_BIN)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.out.resolve() == Path(r"D:\tmp\rism-pdm-hw").resolve():
        print("ERROR: Refusing to touch D:\\tmp\\rism-pdm-hw", file=sys.stderr)
        return 1

    eval_slugs = load_eval_slugs()
    state = RismHarvestState(args.out, limit=args.limit, resume=args.resume)
    print(f"[init] out={args.out} existing={state.count} processed={len(state.processed_sources)}", flush=True)

    t0 = time.time()
    get_clip_classifier()
    print(f"[clip] CPU CLIP classifier ready in {time.time() - t0:.1f}s", flush=True)

    urls = fetch_institution_source_urls("PL-Wn", max_pages=6, rows=100)
    urls = [u for u in urls if u.rstrip("/").split("/")[-1] not in state.processed_sources]
    print(f"[search] PL-Wn: queued {len(urls)} unprocessed manuscript candidates", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(process_polona_source, u, state, args.assembly_bin, eval_slugs, args.max_dim, args.clip_threshold)
            for u in urls
        ]
        for fut in concurrent.futures.as_completed(futs):
            try:
                fut.result()
            except Exception as e:
                state.bump("no_iiif_url", f"worker_exc:{type(e).__name__}")

    summary = state.write_summary()
    total_ver, overlaps = verify_zero_eval_overlap(state.manifest_path)
    print(f"[verify] manifest={state.manifest_path} total={total_ver} eval_overlaps={overlaps}", flush=True)
    print("[finished] summary:\n" + json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
