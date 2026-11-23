#!/usr/bin/env python3
"""Harvest 10,000 truly unlabeled, Public Domain / CC0 sheet music page images.

Materializes real JPEG page images (max_dim <= 1600, Q=88) into
`<out>/images/` plus `<out>/manifest.jsonl` and `<out>/summary.json`.

Sources (100% Public Domain / CC0 / permissive, documented public APIs only):
  1. Local non-eval PD IMSLP manuscript pages (`data-gen/score-corpus/handwritten-500`)
     excluding the 5 held-out `suite31_hw` works.
  2. Internet Archive (`archive.org/advancedsearch.php`):
     - `collection:imslp` (Public Domain IMSLP scores, downloadable PDFs)
     - `subject:"sheet music" AND date:[1700 TO 1927]` (pre-1928 US PD sheet music)
  3. Wikimedia Commons (`commons.wikimedia.org/w/api.php`):
     - Public Domain / CC0 verified sheet music & musical manuscript images.

CPU + network only (zero CUDA/GPU usage). Zero reuse of labeled training corpora
(`unified-sysdata`, `quota-sysdata`, `leland-sysdata`, `big500`) and zero overlap
with `gate20`, `suite31`, or the 5 held-out `suite31_hw` works.

Usage:
    .\\.venv\\Scripts\\python.exe tmp-train/mvd/harvest_unlabeled_10k.py \\
        --target 10000 --out D:\\tmp\\unlabeled-10k-real --workers 8 --resume
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import os

# Hard-disable CUDA/GPU before torch can ever be imported (CPU + network only)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import re
import sys
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pymupdf
import requests
from PIL import Image

USER_AGENT = "sheet-music-converter-research/1.0 (public-domain-omr-dataset; contact: research@local)"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HW500_DIR = REPO_ROOT / "data-gen" / "score-corpus" / "handwritten-500" / "handwritten"
DEFAULT_GATE20_MANIFEST = REPO_ROOT / "data-gen" / "score-corpus" / "gate20" / "gate_manifest.json"
DEFAULT_SUITE31_MANIFEST = REPO_ROOT / "data-gen" / "score-corpus" / "suite31" / "suite31_manifest.json"

# 5 held-out suite31_hw works (strictly excluded everywhere)
HELD_OUT_HW_WORKS = (
    "beethoven-op135",
    "mozart-k464",
    "mozart-k387",
    "mozart-k421",
    "schubert-d810",
)

# Normalized (composer, work_token) pairs and standalone title tokens for eval leakage protection
EVAL_PAIR_EXCLUSIONS: Tuple[Tuple[str, str], ...] = (
    # 5 held-out suite31_hw works
    ("beethoven", "135"),
    ("mozart", "464"),
    ("mozart", "387"),
    ("mozart", "421"),
    ("schubert", "810"),
    ("schubert", "death and the maiden"),
    ("schubert", "tod und das madchen"),
    # gate20 works
    ("brahms", "bitteres zu sagen"),
    ("chabrier", "ballade des gros dindons"),
    ("greger", "plattdeutsche"),
    ("mahler", "fahrenden gesellen"),
    ("mahler", "gluhend messer"),
    ("paradis", "sophia an siegwart"),
    ("parry", "poet s song"),
    ("reichardt", "traurige wanderer"),
    ("reichardt", "kaeuzlein"),
    ("reichardt", "erinnrung zum bach"),
    ("reichardt", "vaters klage"),
    ("reichardt", "die veilchen"),
    ("reichardt", "bergmannslied"),
    ("schroter", "die liebe"),
    ("mayer", "op 14"),
    ("saint georges", "op 1 no 3"),
    # suite31 classical works
    ("bach", "bwv 66"),
    ("joplin", "maple leaf"),
    ("schoenberg", "op 19"),
    ("schonberg", "op 19"),
    ("beethoven", "grosse fuge"),
    ("beethoven", "op 133"),
    ("mozart", "k 155"),
    ("mozart", "k155"),
    ("mozart", "k 458"),
    ("mozart", "k458"),
    ("beethoven", "op 18 no 1"),
    ("haydn", "op 74 no 1"),
    ("chopin", "mazurka op 6"),
    ("schubert", "lindenbaum"),
    ("schumann", "dichterliebe"),
    ("schumann", "op 41 no 1"),
    ("clara schumann", "op 17"),
    ("weber", "concertino"),
    ("verdi", "la donna e mobile"),
    ("handel", "lascia ch io pianga"),
    ("palestrina", "agnus dei"),
)

EVAL_STANDALONE_EXCLUSIONS: Tuple[str, ...] = (
    "beethoven-op135",
    "mozart-k464",
    "mozart-k387",
    "mozart-k421",
    "schubert-d810",
    "ballade des gros dindons",
    "bitteres zu sagen",
    "sophia an siegwart",
    "maple leaf rag",
    "grosse fuge",
    "aloha oe",
    "unified-sysdata",
    "quota-sysdata",
    "leland-sysdata",
    "big500",
    "gate20",
    "suite31",
)

# Allowed PD / CC0 license markers for Wikimedia Commons
PD_LICENSE_MARKERS = (
    "public domain",
    "pd-old",
    "pd-us",
    "pd-art",
    "pd-1923",
    "pd-1996",
    "pd-self",
    "pd-author",
    "cc0",
    "cc-zero",
)

# Symbolic notation extensions: any item containing these is LABELED and must be
# routed to `labeled_finds.jsonl` instead of the unlabeled image pool.
LABELED_EXTENSIONS = (".musicxml", ".mxl", ".mscx", ".mscz", ".krn")


def find_labeled_files(files: Iterable[Dict[str, Any]]) -> List[str]:
    """Return list of filenames in an item that are symbolic notation labels."""
    found: List[str] = []
    for f in files:
        name = str(f.get("name", "")).strip()
        if not name:
            continue
        lname = name.lower()
        if lname.endswith(LABELED_EXTENSIONS):
            found.append(name)
    return found


def normalize_text(text: str) -> str:
    """Lowercase, strip accents, collapse non-alphanumeric to single spaces."""
    s = unicodedata.normalize("NFKD", text or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return s


def load_eval_slugs(
    gate20_path: Optional[Path] = DEFAULT_GATE20_MANIFEST,
    suite31_path: Optional[Path] = DEFAULT_SUITE31_MANIFEST,
) -> Set[str]:
    """Load exact slugs / score IDs from gate20 and suite31 manifests if present."""
    slugs: Set[str] = set()
    if gate20_path and gate20_path.is_file():
        try:
            g = json.loads(gate20_path.read_text(encoding="utf-8"))
            for p in g.get("pages", []):
                for key in ("id", "score"):
                    val = p.get(key)
                    if val:
                        slugs.add(normalize_text(val))
        except Exception:
            pass
    if suite31_path and suite31_path.is_file():
        try:
            s = json.loads(suite31_path.read_text(encoding="utf-8"))
            for p in s.get("pieces", []):
                val = p.get("slug")
                if val:
                    slugs.add(normalize_text(val))
        except Exception:
            pass
    return {x for x in slugs if len(x) >= 4}


def is_eval_excluded(
    text: str,
    item_id: str = "",
    extra_slugs: Optional[Set[str]] = None,
    strict_suite31_hw_only: bool = False,
) -> Tuple[bool, str]:
    """Return (excluded, reason) if text/item_id overlaps held-out eval sets.

    When `strict_suite31_hw_only=True` (used for the local `handwritten-500`
    directory whose 365 non-eval interior pages are explicitly requested),
    only the 5 held-out `suite31_hw` works are blocked.
    """
    raw_combined = f"{item_id} {text}".lower()
    norm = normalize_text(raw_combined)

    # Always block the 5 held-out suite31_hw works
    for hw in HELD_OUT_HW_WORKS:
        if hw in raw_combined or normalize_text(hw) in norm:
            return True, f"held_out_hw:{hw}"
    for comp, tok in EVAL_PAIR_EXCLUSIONS[:7]:
        if comp in norm and re.search(rf"\b{re.escape(tok)}\b", norm):
            return True, f"held_out_hw_pair:{comp}+{tok}"

    if strict_suite31_hw_only:
        return False, ""

    # Block standalone eval identifiers/titles
    for tok in EVAL_STANDALONE_EXCLUSIONS:
        ntok = normalize_text(tok)
        if tok in raw_combined or (ntok and re.search(rf"\b{re.escape(ntok)}\b", norm)):
            return True, f"eval_standalone:{tok}"

    # Block composer + work pairs
    for comp, tok in EVAL_PAIR_EXCLUSIONS[7:]:
        if comp in norm and re.search(rf"\b{re.escape(tok)}\b", norm):
            return True, f"eval_pair:{comp}+{tok}"

    # Block exact slugs from gate20/suite31 manifests
    if extra_slugs:
        for slug in extra_slugs:
            if re.search(rf"\b{re.escape(slug)}\b", norm):
                return True, f"eval_slug:{slug}"

    return False, ""


CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
CLIP_MUSIC_THRESHOLD = 0.65
CLIP_PROMPTS: Tuple[str, ...] = (
    # music_sheet (indices 0, 1)
    "sheet music score with musical staves and notes",
    "handwritten musical manuscript score with staff lines",
    # text_page (indices 2, 3, 4)
    "a book page of printed prose text paragraphs",
    "a table of contents, preface, or index page with words",
    "a handwritten letter, cursive notes, or archival document page",
    # cover_or_blank (indices 5, 6, 7)
    "a vintage book cover, title page, frontispiece, or decorative illustration",
    "a library stamp, archival label, cover sheet, or blank flyleaf page",
    "a blank empty white or aged paper page",
)


class ClipPageClassifier:
    """Deterministic, zero-cost, CPU-only 3-class visual page classifier.

    Classes: `music_sheet`, `text_page`, `cover_or_blank`.
    Handles printed, aged, tilted, and handwritten sepia manuscripts cleanly.
    """

    def __init__(self, model_id: str = CLIP_MODEL_ID):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._torch = torch
        self._lock = threading.Lock()
        self.model_id = model_id
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to("cpu").eval()

        # Pre-compute normalized text features once on CPU
        with torch.no_grad():
            text_inputs = self.processor(text=list(CLIP_PROMPTS), return_tensors="pt", padding=True)
            t_out = self.model.get_text_features(**text_inputs)
            if not isinstance(t_out, torch.Tensor):
                t_out = t_out.pooler_output
            t_out = t_out / t_out.norm(dim=-1, keepdim=True)
            self.text_features = t_out.cpu()
            self.logit_scale = self.model.logit_scale.exp().detach().cpu()

    def classify_batch(
        self,
        images: Sequence[Image.Image],
        threshold: float = CLIP_MUSIC_THRESHOLD,
    ) -> List[Tuple[bool, Dict[str, Any]]]:
        """Classify a batch of PIL images on CPU; returns [(is_music, info_dict), ...]."""
        if not images:
            return []
        torch = self._torch
        rgb_imgs = [im.convert("RGB") for im in images]
        inputs = self.processor(images=rgb_imgs, return_tensors="pt")
        pixel_values = inputs["pixel_values"]

        with self._lock, torch.no_grad():
            img_out = self.model.get_image_features(pixel_values=pixel_values)
            if not isinstance(img_out, torch.Tensor):
                img_out = img_out.pooler_output
            img_feats = img_out / img_out.norm(dim=-1, keepdim=True)
            logits = self.logit_scale * (img_feats @ self.text_features.T)
            probs = logits.softmax(dim=-1).cpu().numpy()

        results: List[Tuple[bool, Dict[str, Any]]] = []
        for p in probs:
            p_music = float(p[0] + p[1])
            p_text = float(p[2] + p[3] + p[4])
            p_cover = float(p[5] + p[6] + p[7])
            if p_music >= max(p_text, p_cover):
                page_class = "music_sheet"
            elif p_text >= p_cover:
                page_class = "text_page"
            else:
                page_class = "cover_or_blank"
            ok = page_class == "music_sheet" and p_music >= threshold
            results.append(
                (
                    ok,
                    {
                        "page_class": page_class,
                        "p_music": round(p_music, 4),
                        "p_text": round(p_text, 4),
                        "p_cover": round(p_cover, 4),
                    },
                )
            )
        return results


_CLIP_INIT_LOCK = threading.Lock()
_CLIP_CLASSIFIER: Optional[ClipPageClassifier] = None


def get_clip_classifier() -> ClipPageClassifier:
    """Thread-safe lazy singleton for the CPU-only CLIP page classifier."""
    global _CLIP_CLASSIFIER
    if _CLIP_CLASSIFIER is not None:
        return _CLIP_CLASSIFIER
    with _CLIP_INIT_LOCK:
        if _CLIP_CLASSIFIER is None:
            _CLIP_CLASSIFIER = ClipPageClassifier(CLIP_MODEL_ID)
        return _CLIP_CLASSIFIER


def is_music_page(
    img: Image.Image,
    threshold: float = CLIP_MUSIC_THRESHOLD,
) -> Tuple[bool, Dict[str, Any]]:
    """Classify a single page with CPU CLIP 3-class visual classifier.

    Accepts printed AND handwritten/sepia/tilted sheet music (`music_sheet`,
    `p_music >= threshold`) and rejects `text_page` and `cover_or_blank`.
    """
    w, h = img.size
    if w < 250 or h < 250:
        return False, {"page_class": "cover_or_blank", "p_music": 0.0, "w": float(w), "h": float(h)}
    aspect = h / float(w)
    if aspect < 0.45 or aspect > 2.6:
        return False, {"page_class": "cover_or_blank", "p_music": 0.0, "aspect": round(aspect, 3)}

    clf = get_clip_classifier()
    ok, stats = clf.classify_batch([img], threshold=threshold)[0]
    return ok, stats


def classify_pages_batch(
    images: Sequence[Image.Image],
    threshold: float = CLIP_MUSIC_THRESHOLD,
    batch_size: int = 32,
) -> List[Tuple[bool, Dict[str, Any]]]:
    """Batched CPU CLIP classification for high throughput (~25-40 img/s on CPU)."""
    out: List[Optional[Tuple[bool, Dict[str, Any]]]] = [None] * len(images)
    valid_idx: List[int] = []
    valid_imgs: List[Image.Image] = []
    for i, im in enumerate(images):
        w, h = im.size
        aspect = h / float(max(1, w))
        if w < 250 or h < 250 or aspect < 0.45 or aspect > 2.6:
            out[i] = (False, {"page_class": "cover_or_blank", "p_music": 0.0})
        else:
            valid_idx.append(i)
            valid_imgs.append(im)

    if valid_imgs:
        clf = get_clip_classifier()
        for start in range(0, len(valid_imgs), batch_size):
            chunk = valid_imgs[start : start + batch_size]
            res = clf.classify_batch(chunk, threshold=threshold)
            for j, r in enumerate(res):
                out[valid_idx[start + j]] = r

    return [x if x is not None else (False, {"page_class": "cover_or_blank", "p_music": 0.0}) for x in out]


def resize_and_encode_jpeg(img: Image.Image, max_dim: int = 1600, quality: int = 88) -> Tuple[bytes, int, int]:
    """Resize image so max(w, h) <= max_dim and encode as RGB JPEG."""
    rgb = img.convert("RGB")
    w, h = rgb.size
    if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        rgb = rgb.resize((nw, nh), Image.Resampling.LANCZOS)
        w, h = nw, nh
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue(), w, h


def safe_slug(text: str, max_len: int = 64) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_")
    return (s[:max_len] or "item").lower()


def http_get_with_retry(
    session: requests.Session,
    url: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: float = 25.0,
    retries: int = 3,
    stream: bool = False,
) -> Optional[requests.Response]:
    """Polite GET with exponential backoff; skips cleanly on 403/404/429/5xx exhaustion."""
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=timeout, stream=stream)
            if r.status_code == 200:
                return r
            if r.status_code in (401, 403, 404, 410, 451):
                r.close()
                return None
            if r.status_code == 429:
                r.close()
                time.sleep(min(8.0, 2.0 * (attempt + 1)))
                continue
            r.close()
            time.sleep(1.0 * (attempt + 1))
        except requests.RequestException:
            time.sleep(1.0 * (attempt + 1))
    return None


class HarvestState:
    """Thread-safe materialized dataset state with resume & dedup."""

    def __init__(self, out_dir: Path, target: int, resume: bool = True):
        self.out_dir = out_dir
        self.images_dir = out_dir / "images"
        self.manifest_path = out_dir / "manifest.jsonl"
        self.labeled_finds_path = out_dir / "labeled_finds.jsonl"
        self.processed_path = out_dir / "processed_items.txt"
        self.summary_path = out_dir / "summary.json"
        self.target = target
        self.lock = threading.Lock()
        self.start_time = time.time()

        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.seen_hashes: Set[str] = set()
        self.processed_items: Set[str] = set()
        self.count = 0
        self.by_source: Dict[str, int] = {}
        self.by_collection: Dict[str, int] = {}
        self.rejected_non_music = 0
        self.rejected_eval = 0
        self.rejected_dup = 0
        self.labeled_finds_count = 0
        self.failed_items = 0

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
                    img_p = Path(rec.get("image_path", ""))
                    if not img_p.is_file():
                        continue
                    sha = rec.get("sha256", "")
                    if sha:
                        self.seen_hashes.add(sha)
                    self.count += 1
                    src = rec.get("source", "unknown")
                    col = rec.get("collection", "unknown")
                    self.by_source[src] = self.by_source.get(src, 0) + 1
                    self.by_collection[col] = self.by_collection.get(col, 0) + 1
        elif not resume:
            for p in (self.manifest_path, self.processed_path):
                if p.exists():
                    p.unlink()

        if resume and self.processed_path.is_file():
            with self.processed_path.open("r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s:
                        self.processed_items.add(s)

    def is_done(self) -> bool:
        with self.lock:
            return self.count >= self.target

    def current_count(self) -> int:
        with self.lock:
            return self.count

    def mark_item_processed(self, item_key: str) -> None:
        with self.lock:
            if item_key not in self.processed_items:
                self.processed_items.add(item_key)
                with self.processed_path.open("a", encoding="utf-8") as f:
                    f.write(item_key + "\n")

    def is_item_processed(self, item_key: str) -> bool:
        with self.lock:
            return item_key in self.processed_items

    def add_rejected(self, kind: str, n: int = 1) -> None:
        with self.lock:
            if kind == "non_music":
                self.rejected_non_music += n
            elif kind == "eval":
                self.rejected_eval += n
            elif kind == "dup":
                self.rejected_dup += n
            elif kind == "failed":
                self.failed_items += n

    def record_labeled_find(
        self,
        source: str,
        collection: str,
        item_id: str,
        title: str,
        url: str,
        labeled_files: List[str],
    ) -> None:
        rec = {
            "source": source,
            "collection": collection,
            "item_id": item_id,
            "title": title,
            "url": url,
            "labeled_files": labeled_files,
        }
        with self.lock:
            self.labeled_finds_count += 1
            with self.labeled_finds_path.open("a", encoding="utf-8") as lf:
                lf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def try_save_image(
        self,
        jpeg_bytes: bytes,
        width: int,
        height: int,
        filename: str,
        source: str,
        collection: str,
        item_id: str,
        title: str,
        license_str: str,
        url: str,
        page_index: int,
        filter_stats: Dict[str, float],
    ) -> bool:
        sha = hashlib.sha256(jpeg_bytes).hexdigest()
        with self.lock:
            if self.count >= self.target:
                return False
            if sha in self.seen_hashes:
                self.rejected_dup += 1
                return False

            out_path = self.images_dir / filename
            if out_path.exists():
                stem = out_path.stem
                out_path = self.images_dir / f"{stem}_{sha[:8]}.jpg"

            tmp_path = out_path.with_suffix(".jpg.tmp")
            tmp_path.write_bytes(jpeg_bytes)
            tmp_path.replace(out_path)

            self.seen_hashes.add(sha)
            self.count += 1
            self.by_source[source] = self.by_source.get(source, 0) + 1
            self.by_collection[collection] = self.by_collection.get(collection, 0) + 1

            rec = {
                "id": out_path.stem,
                "image_path": str(out_path),
                "source": source,
                "collection": collection,
                "item_id": item_id,
                "title": title,
                "license": license_str,
                "url": url,
                "page_index": page_index,
                "width": width,
                "height": height,
                "sha256": sha,
                "page_class": filter_stats.get("page_class", "music_sheet"),
                "p_music": filter_stats.get("p_music", 1.0),
                "filter_stats": filter_stats,
            }
            with self.manifest_path.open("a", encoding="utf-8") as mf:
                mf.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if self.count % 100 == 0 or self.count == self.target:
                elapsed = max(0.1, time.time() - self.start_time)
                rate = self.count / elapsed
                print(
                    f"[harvest] {self.count}/{self.target} images "
                    f"({rate:.1f} img/s, elapsed {elapsed:.0f}s) | "
                    f"sources={dict(self.by_source)} | "
                    f"rej_non_music={self.rejected_non_music} rej_eval={self.rejected_eval} rej_dup={self.rejected_dup}",
                    flush=True,
                )
                self._write_summary_locked(leaked_eval=0)
            return True

    def _write_summary_locked(self, leaked_eval: int) -> Dict[str, Any]:
        summary = {
            "out_dir": str(self.out_dir),
            "images_dir": str(self.images_dir),
            "manifest_path": str(self.manifest_path),
            "labeled_finds_path": str(self.labeled_finds_path),
            "target": self.target,
            "total_images": self.count,
            "by_source": dict(self.by_source),
            "by_collection": dict(self.by_collection),
            "rejected_non_music": self.rejected_non_music,
            "rejected_eval": self.rejected_eval,
            "rejected_dup": self.rejected_dup,
            "labeled_finds_routed": self.labeled_finds_count,
            "failed_items": self.failed_items,
            "leaked_eval": leaked_eval,
            "elapsed_sec": round(time.time() - self.start_time, 1),
        }
        tmp = self.summary_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.summary_path)
        return summary

    def finalize_and_audit(self, extra_slugs: Set[str]) -> Dict[str, Any]:
        """Re-scan manifest.jsonl to verify leaked_eval == 0 and exact on-disk file count."""
        leaked = 0
        on_disk = 0
        if self.manifest_path.is_file():
            with self.manifest_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if Path(rec["image_path"]).is_file():
                        on_disk += 1
                    strict_hw = rec.get("source") == "pd_manuscripts_hw500"
                    excl, _ = is_eval_excluded(
                        rec.get("title", ""),
                        rec.get("item_id", ""),
                        extra_slugs=extra_slugs,
                        strict_suite31_hw_only=strict_hw,
                    )
                    if excl:
                        leaked += 1
        with self.lock:
            self.count = on_disk
            return self._write_summary_locked(leaked_eval=leaked)


def harvest_local_hw500(
    state: HarvestState,
    hw500_dir: Path,
    max_dim: int,
    extra_slugs: Set[str],
) -> int:
    """Ingest truly unlabeled non-eval PD IMSLP manuscript pages from handwritten-500."""
    if not hw500_dir.is_dir():
        return 0
    added = 0
    for work_dir in sorted(hw500_dir.iterdir()):
        if state.is_done():
            break
        if not work_dir.is_dir():
            continue
        work_name = work_dir.name
        item_key = f"hw500:{work_name}"
        excl, reason = is_eval_excluded(work_name, work_name, extra_slugs=extra_slugs, strict_suite31_hw_only=True)
        if excl:
            state.add_rejected("eval")
            continue
        if state.is_item_processed(item_key):
            continue

        img_files = sorted(
            p for p in work_dir.iterdir() if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")
        )
        for idx, img_path in enumerate(img_files):
            if state.is_done():
                break
            try:
                with Image.open(img_path) as im:
                    ok, stats = is_music_page(im)
                    if not ok:
                        state.add_rejected("non_music")
                        continue
                    jpeg_bytes, w, h = resize_and_encode_jpeg(im, max_dim=max_dim)
            except Exception:
                state.add_rejected("failed")
                continue

            fname = f"hw500_{safe_slug(work_name)}_p{idx:04d}.jpg"
            saved = state.try_save_image(
                jpeg_bytes=jpeg_bytes,
                width=w,
                height=h,
                filename=fname,
                source="pd_manuscripts_hw500",
                collection="imslp_pd_manuscripts",
                item_id=work_name,
                title=f"PD Manuscript {work_name}",
                license_str="Public Domain (IMSLP)",
                url=f"local://handwritten-500/{work_name}/{img_path.name}",
                page_index=idx,
                filter_stats=stats,
            )
            if saved:
                added += 1
        state.mark_item_processed(item_key)
    return added


def fetch_ia_candidates(
    session: requests.Session,
    query: str,
    collection_label: str,
    max_items: int,
    extra_slugs: Set[str],
    rows_per_page: int = 200,
) -> List[Dict[str, Any]]:
    """Query Internet Archive advancedsearch API for candidate PD PDF items."""
    out: List[Dict[str, Any]] = []
    page = 1
    while len(out) < max_items:
        params = {
            "q": query,
            "fl[]": ["identifier", "title", "creator", "date", "downloads"],
            "sort[]": "downloads desc",
            "rows": min(rows_per_page, max_items - len(out) + 50),
            "page": page,
            "output": "json",
        }
        r = http_get_with_retry(session, "https://archive.org/advancedsearch.php", params=params, timeout=25.0)
        if r is None:
            break
        try:
            docs = r.json().get("response", {}).get("docs", [])
        except Exception:
            break
        if not docs:
            break
        for doc in docs:
            ident = str(doc.get("identifier", "")).strip()
            title = doc.get("title", "")
            if isinstance(title, list):
                title = " ".join(str(x) for x in title)
            creator = doc.get("creator", "")
            if isinstance(creator, list):
                creator = " ".join(str(x) for x in creator)
            if not ident:
                continue
            excl, _ = is_eval_excluded(f"{creator} {title}", ident, extra_slugs=extra_slugs)
            if excl:
                continue
            out.append(
                {
                    "identifier": ident,
                    "title": str(title).strip() or ident,
                    "creator": str(creator).strip(),
                    "date": str(doc.get("date", "")),
                    "collection_label": collection_label,
                }
            )
            if len(out) >= max_items:
                break
        page += 1
        if page > 50:
            break
    return out


def pick_best_pdf(files: List[Dict[str, Any]], max_pdf_bytes: int = 35_000_000) -> Optional[Tuple[str, int]]:
    """Choose the best downloadable PDF from an IA item's file metadata."""
    candidates: List[Tuple[int, str, int]] = []
    for f in files:
        name = str(f.get("name", ""))
        if not name.lower().endswith(".pdf"):
            continue
        if f.get("private") in (True, "true", "1"):
            continue
        try:
            size = int(f.get("size", 0))
        except (TypeError, ValueError):
            size = 0
        if size < 40_000 or size > max_pdf_bytes:
            continue
        lname = name.lower()
        # Prefer standard PDF over _bw / _text / _abbyy derivatives
        prio = 0
        if "_text.pdf" in lname or "ocr" in lname:
            prio = 3
        elif "_bw.pdf" in lname:
            prio = 2
        else:
            prio = 1
        candidates.append((prio, name, size))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[2]))
    _, best_name, best_size = candidates[0]
    return best_name, best_size


def process_ia_item(
    item: Dict[str, Any],
    state: HarvestState,
    max_dim: int,
    max_pages_per_item: int,
    max_pdf_bytes: int,
    bucket_limit: Optional[int] = None,
) -> int:
    """Download one IA PDF into memory, rasterize & filter pages, save JPEGs."""
    if state.is_done():
        return 0
    col = item["collection_label"]
    if bucket_limit is not None:
        with state.lock:
            if state.by_collection.get(col, 0) >= bucket_limit:
                return 0

    ident = item["identifier"]
    item_key = f"ia:{ident}"
    if state.is_item_processed(item_key):
        return 0

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    # 1. Fetch file list metadata
    mr = http_get_with_retry(session, f"https://archive.org/metadata/{ident}/files", timeout=20.0, retries=2)
    if mr is None:
        state.add_rejected("failed")
        state.mark_item_processed(item_key)
        return 0
    try:
        files = mr.json().get("result", [])
    except Exception:
        state.add_rejected("failed")
        state.mark_item_processed(item_key)
        return 0

    # Standing rule: NEVER place items with MusicXML/.mxl/.mscx/.mscz/.krn labels
    # into the unlabeled pool; route them to labeled_finds.jsonl instead.
    labeled_files = find_labeled_files(files)
    if labeled_files:
        state.record_labeled_find(
            source="internet_archive",
            collection=col,
            item_id=ident,
            title=item["title"],
            url=f"https://archive.org/details/{ident}",
            labeled_files=labeled_files,
        )
        state.mark_item_processed(item_key)
        session.close()
        return 0

    picked = pick_best_pdf(files, max_pdf_bytes=max_pdf_bytes)
    if picked is None:
        state.mark_item_processed(item_key)
        return 0
    pdf_name, _ = picked

    # 2. Download PDF bytes (in memory; never leaves temp PDFs on disk)
    dl_url = f"https://archive.org/download/{ident}/{requests.utils.quote(pdf_name)}"
    pr = http_get_with_retry(session, dl_url, timeout=45.0, retries=2)
    if pr is None or len(pr.content) < 40_000:
        state.add_rejected("failed")
        state.mark_item_processed(item_key)
        return 0

    saved_count = 0
    try:
        doc = pymupdf.open(stream=pr.content, filetype="pdf")
    except Exception:
        state.add_rejected("failed")
        state.mark_item_processed(item_key)
        return 0

    try:
        n_pages = len(doc)
        # If a score has many pages, stride evenly so we stay diverse within max_pages_per_item
        if n_pages <= max_pages_per_item:
            page_indices = list(range(n_pages))
        else:
            step = n_pages / float(max_pages_per_item)
            page_indices = sorted({min(n_pages - 1, int(round(i * step))) for i in range(max_pages_per_item)})

        license_str = (
            "Public Domain (IMSLP / Internet Archive)"
            if col == "ia_imslp"
            else "Public Domain (Pre-1928 / Internet Archive)"
        )
        slug = safe_slug(ident, max_len=48)

        rendered: List[Tuple[int, Image.Image]] = []
        for pidx in page_indices:
            if state.is_done():
                break
            if bucket_limit is not None:
                with state.lock:
                    if state.by_collection.get(col, 0) >= bucket_limit:
                        break
            try:
                page = doc[pidx]
                rect = page.rect
                longest_pt = max(rect.width, rect.height, 1.0)
                # Rasterize directly near target max_dim (<= 1600 px, max ~180 DPI)
                scale = min(180.0 / 72.0, max_dim / longest_pt)
                scale = max(0.5, scale)
                pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
                im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                rendered.append((pidx, im))
            except Exception:
                continue

        if rendered:
            verdicts = classify_pages_batch([im for _, im in rendered], batch_size=32)
            for (pidx, im), (ok, stats) in zip(rendered, verdicts):
                if state.is_done():
                    break
                if bucket_limit is not None:
                    with state.lock:
                        if state.by_collection.get(col, 0) >= bucket_limit:
                            break
                if not ok:
                    state.add_rejected("non_music")
                    continue

                jpeg_bytes, w, h = resize_and_encode_jpeg(im, max_dim=max_dim)
                fname = f"{col}_{slug}_p{pidx:04d}.jpg"
                saved = state.try_save_image(
                    jpeg_bytes=jpeg_bytes,
                    width=w,
                    height=h,
                    filename=fname,
                    source="internet_archive",
                    collection=col,
                    item_id=ident,
                    title=item["title"],
                    license_str=license_str,
                    url=dl_url,
                    page_index=pidx,
                    filter_stats=stats,
                )
                if saved:
                    saved_count += 1
    finally:
        doc.close()
        session.close()

    state.mark_item_processed(item_key)
    return saved_count


def harvest_wikimedia_commons(
    state: HarvestState,
    max_images: int,
    max_dim: int,
    extra_slugs: Set[str],
    workers: int = 6,
) -> int:
    """Harvest verified Public Domain / CC0 sheet music scans from Wikimedia Commons API."""
    if max_images <= 0 or state.is_done():
        return 0

    categories = [
        "Category:19th-century sheet music",
        "Category:18th-century sheet music",
        "Category:Sheet music",
        "Category:Musical manuscripts",
        "Category:Printed music",
    ]
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    file_titles: List[str] = []
    seen_titles: Set[str] = set()
    for cat in categories:
        cmcontinue = None
        for _ in range(8):
            if len(file_titles) >= max_images * 3:
                break
            params: Dict[str, Any] = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": cat,
                "cmtype": "file",
                "cmlimit": 100,
                "format": "json",
            }
            if cmcontinue:
                params["cmcontinue"] = cmcontinue
            r = http_get_with_retry(session, "https://commons.wikimedia.org/w/api.php", params=params, timeout=20.0)
            if r is None:
                break
            data = r.json()
            for cm in data.get("query", {}).get("categorymembers", []):
                t = cm.get("title", "")
                if not t or t in seen_titles:
                    continue
                lt = t.lower()
                if not lt.endswith((".jpg", ".jpeg", ".png")):
                    continue
                excl, _ = is_eval_excluded(t, t, extra_slugs=extra_slugs)
                if excl:
                    state.add_rejected("eval")
                    continue
                seen_titles.add(t)
                file_titles.append(t)
            cmcontinue = data.get("continue", {}).get("cmcontinue")
            if not cmcontinue:
                break

    # Fetch imageinfo + extmetadata in batches of 40 to verify PD/CC0 license
    pd_items: List[Dict[str, Any]] = []
    for i in range(0, len(file_titles), 40):
        if len(pd_items) >= max_images * 2:
            break
        batch = file_titles[i : i + 40]
        params = {
            "action": "query",
            "titles": "|".join(batch),
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata",
            "iiurlwidth": max_dim,
            "format": "json",
        }
        r = http_get_with_retry(session, "https://commons.wikimedia.org/w/api.php", params=params, timeout=25.0)
        if r is None:
            continue
        pages = r.json().get("query", {}).get("pages", {})
        for pinfo in pages.values():
            title = pinfo.get("title", "")
            iis = pinfo.get("imageinfo", [])
            if not iis:
                continue
            ii = iis[0]
            meta = ii.get("extmetadata", {})
            lic = str(meta.get("LicenseShortName", {}).get("value", "")).strip()
            lic_low = lic.lower()
            if not any(m in lic_low for m in PD_LICENSE_MARKERS):
                continue
            url = ii.get("thumburl") or ii.get("url")
            if not url:
                continue
            pd_items.append({"title": title, "url": url, "license": lic or "Public Domain"})

    session.close()
    added = 0
    wc_lock = threading.Lock()

    def _download_one(it: Dict[str, Any]) -> int:
        nonlocal added
        if state.is_done():
            return 0
        with wc_lock:
            if added >= max_images:
                return 0
        item_key = f"wc:{it['title']}"
        if state.is_item_processed(item_key):
            return 0
        s = requests.Session()
        s.headers.update({"User-Agent": USER_AGENT})
        try:
            r = http_get_with_retry(s, it["url"], timeout=25.0, retries=2)
            if r is None or len(r.content) < 8_000:
                state.add_rejected("failed")
                state.mark_item_processed(item_key)
                return 0
            im = Image.open(io.BytesIO(r.content))
            ok, stats = is_music_page(im)
            if not ok:
                state.add_rejected("non_music")
                state.mark_item_processed(item_key)
                return 0
            jpeg_bytes, w, h = resize_and_encode_jpeg(im, max_dim=max_dim)
            slug = safe_slug(it["title"].replace("File:", ""), max_len=56)
            saved = state.try_save_image(
                jpeg_bytes=jpeg_bytes,
                width=w,
                height=h,
                filename=f"wc_{slug}.jpg",
                source="wikimedia_commons",
                collection="wikimedia_commons_pd",
                item_id=it["title"],
                title=it["title"],
                license_str=it["license"],
                url=it["url"],
                page_index=0,
                filter_stats=stats,
            )
            state.mark_item_processed(item_key)
            if saved:
                with wc_lock:
                    added += 1
                return 1
            return 0
        except Exception:
            state.add_rejected("failed")
            state.mark_item_processed(item_key)
            return 0
        finally:
            s.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, 6)) as ex:
        list(ex.map(_download_one, pd_items))
    return added


def cleanup_existing_with_clip(
    out_dir: Path,
    workers: int = 8,
    batch_size: int = 32,
    threshold: float = CLIP_MUSIC_THRESHOLD,
) -> Dict[str, Any]:
    """Audit all existing images in `<out_dir>/images/` with CPU CLIP 3-class classifier,
    purging `text_page` and `cover_or_blank` (or `p_music < threshold`) while keeping
    both printed and handwritten `music_sheet` pages, and rewriting `manifest.jsonl` + `summary.json`.
    """
    get_clip_classifier()
    images_dir = out_dir / "images"
    manifest_path = out_dir / "manifest.jsonl"
    summary_path = out_dir / "summary.json"
    extra_slugs = load_eval_slugs()

    records: List[Dict[str, Any]] = []
    if manifest_path.is_file():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    t0 = time.time()
    kept_records: List[Dict[str, Any]] = []
    kept_paths: Set[Path] = set()
    purged_by_class: Dict[str, int] = {"text_page": 0, "cover_or_blank": 0, "low_conf_music": 0, "unreadable": 0}
    purged_missing = 0

    def _load_img(rec: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Path], Optional[Image.Image]]:
        p = Path(rec.get("image_path", ""))
        if not p.is_file():
            return rec, None, None
        try:
            with Image.open(p) as im:
                rgb = im.convert("RGB")
                rgb.load()
            return rec, p, rgb
        except Exception:
            return rec, p, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, 6)) as loader:
        for start in range(0, len(records), batch_size):
            chunk_recs = records[start : start + batch_size]
            loaded = list(loader.map(_load_img, chunk_recs))

            valid_items: List[Tuple[Dict[str, Any], Path, Image.Image]] = []
            for rec, p, im in loaded:
                if p is None:
                    purged_missing += 1
                elif im is None:
                    purged_by_class["unreadable"] += 1
                    try:
                        p.unlink()
                    except OSError:
                        pass
                else:
                    valid_items.append((rec, p, im))

            if valid_items:
                verdicts = classify_pages_batch(
                    [im for _, _, im in valid_items], threshold=threshold, batch_size=batch_size
                )
                for (rec, p, _), (ok, stats) in zip(valid_items, verdicts):
                    if ok:
                        rec["page_class"] = stats["page_class"]
                        rec["p_music"] = stats["p_music"]
                        fs = dict(rec.get("filter_stats", {}))
                        fs.update(stats)
                        rec["filter_stats"] = fs
                        kept_records.append(rec)
                        kept_paths.add(p.resolve())
                    else:
                        cls = stats.get("page_class", "cover_or_blank")
                        if cls == "music_sheet":
                            purged_by_class["low_conf_music"] += 1
                        elif cls in purged_by_class:
                            purged_by_class[cls] += 1
                        else:
                            purged_by_class["cover_or_blank"] += 1
                        try:
                            p.unlink()
                        except OSError:
                            pass

            done_n = min(len(records), start + batch_size)
            if done_n % 512 == 0 or done_n == len(records):
                elapsed = max(0.1, time.time() - t0)
                print(
                    f"[cleanup-clip] audited {done_n}/{len(records)} "
                    f"({done_n/elapsed:.1f} img/s, {elapsed:.1f}s): "
                    f"kept={len(kept_records)}, purged={purged_by_class}",
                    flush=True,
                )

    # Remove any orphan .jpg not in kept_paths
    orphans_removed = 0
    if images_dir.is_dir():
        for p in images_dir.glob("*.jpg"):
            if p.resolve() not in kept_paths:
                try:
                    p.unlink()
                    orphans_removed += 1
                except OSError:
                    pass

    # Rewrite manifest.jsonl atomically
    tmp_m = manifest_path.with_suffix(".jsonl.tmp")
    with tmp_m.open("w", encoding="utf-8") as mf:
        for rec in kept_records:
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp_m.replace(manifest_path)

    # Recompute summary.json
    by_source: Dict[str, int] = {}
    by_collection: Dict[str, int] = {}
    leaked = 0
    for rec in kept_records:
        src = rec.get("source", "unknown")
        col = rec.get("collection", "unknown")
        by_source[src] = by_source.get(src, 0) + 1
        by_collection[col] = by_collection.get(col, 0) + 1
        excl, _ = is_eval_excluded(rec.get("title", ""), rec.get("item_id", ""), extra_slugs=extra_slugs)
        if excl:
            leaked += 1

    total_purged = sum(purged_by_class.values())
    summary = {
        "out_dir": str(out_dir),
        "images_dir": str(images_dir),
        "manifest_path": str(manifest_path),
        "labeled_finds_path": str(out_dir / "labeled_finds.jsonl"),
        "classifier": CLIP_MODEL_ID,
        "p_music_threshold": threshold,
        "target": 10000,
        "total_images": len(kept_records),
        "by_source": by_source,
        "by_collection": by_collection,
        "cleanup_audited": len(records),
        "cleanup_kept": len(kept_records),
        "cleanup_purged_total": total_purged,
        "cleanup_purged_by_class": purged_by_class,
        "cleanup_missing": purged_missing,
        "cleanup_orphans_removed": orphans_removed,
        "leaked_eval": leaked,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    tmp_s = summary_path.with_suffix(".json.tmp")
    tmp_s.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_s.replace(summary_path)
    print(f"[cleanup-clip] DONE: {json.dumps(summary, indent=2)}", flush=True)
    return summary


def run_harvest(
    out_dir: Path,
    target: int = 10000,
    workers: int = 8,
    max_dim: int = 1600,
    max_pages_per_item: int = 40,
    max_pdf_mb: int = 35,
    resume: bool = True,
    hw500_dir: Path = DEFAULT_HW500_DIR,
    include_local_hw500: bool = False,
    wc_quota: int = 250,
    ia_sheetmusic_quota: int = 2200,
) -> Dict[str, Any]:
    get_clip_classifier()
    extra_slugs = load_eval_slugs()
    state = HarvestState(out_dir=out_dir, target=target, resume=resume)
    print(
        f"[harvest] starting: target={target}, existing={state.current_count()}, "
        f"workers={workers}, out={out_dir}, eval_slugs={len(extra_slugs)}",
        flush=True,
    )

    # 1. Optional local PD manuscripts (off by default to keep pool 100% external & unlabeled)
    if include_local_hw500 and not state.is_done() and hw500_dir.is_dir():
        n_hw = harvest_local_hw500(state, hw500_dir=hw500_dir, max_dim=max_dim, extra_slugs=extra_slugs)
        print(f"[harvest] local PD manuscripts added={n_hw}, total={state.current_count()}/{target}", flush=True)

    # 2. Wikimedia Commons verified PD/CC0 sheet music
    existing_wc = state.by_source.get("wikimedia_commons", 0)
    wc_rem = max(0, wc_quota - existing_wc)
    if not state.is_done() and wc_rem > 0:
        n_wc = harvest_wikimedia_commons(
            state, max_images=min(wc_rem, target - state.current_count()),
            max_dim=max_dim, extra_slugs=extra_slugs, workers=min(workers, 6),
        )
        print(f"[harvest] Wikimedia Commons added={n_wc}, total={state.current_count()}/{target}", flush=True)

    # 3. Internet Archive: pre-1928 PD sheet music + IMSLP PD scores
    if not state.is_done():
        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})

        sm_items = fetch_ia_candidates(
            session,
            query='subject:"sheet music" AND date:[1700 TO 1927] AND mediatype:texts AND format:PDF AND -collection:inlibrary',
            collection_label="ia_sheetmusic_pre1928",
            max_items=600,
            extra_slugs=extra_slugs,
        )
        imslp_items = fetch_ia_candidates(
            session,
            query="collection:imslp AND mediatype:texts AND format:PDF AND -collection:inlibrary",
            collection_label="ia_imslp",
            max_items=1600,
            extra_slugs=extra_slugs,
        )
        session.close()
        print(
            f"[harvest] IA candidates fetched: pre1928_sheetmusic={len(sm_items)}, imslp={len(imslp_items)}",
            flush=True,
        )

        max_pdf_bytes = max_pdf_mb * 1_000_000

        # Interleave pre-1928 sheet music & IMSLP so both collections populate smoothly
        work_queue: List[Tuple[Dict[str, Any], Optional[int]]] = []
        i_sm, i_im = 0, 0
        while i_sm < len(sm_items) or i_im < len(imslp_items):
            if i_sm < len(sm_items):
                work_queue.append((sm_items[i_sm], ia_sheetmusic_quota))
                i_sm += 1
            for _ in range(2):
                if i_im < len(imslp_items):
                    work_queue.append((imslp_items[i_im], None))
                    i_im += 1

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futures: List[concurrent.futures.Future] = []
            for item, b_lim in work_queue:
                if state.is_done():
                    break
                futures.append(
                    ex.submit(
                        process_ia_item,
                        item,
                        state,
                        max_dim,
                        max_pages_per_item,
                        max_pdf_bytes,
                        b_lim,
                    )
                )
                # Keep bounded in-flight queue so we stop promptly when target is reached
                if len(futures) >= workers * 4:
                    done, pending = concurrent.futures.wait(
                        futures, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    futures = list(pending)
                    if state.is_done():
                        for f in futures:
                            f.cancel()
                        break
            concurrent.futures.wait(futures)

    summary = state.finalize_and_audit(extra_slugs=extra_slugs)
    print(f"[harvest] DONE: {json.dumps(summary, indent=2)}", flush=True)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Harvest 10k real unlabeled PD/CC0 sheet music images.")
    ap.add_argument("--target", type=int, default=10000, help="Target number of materialized JPEG images.")
    ap.add_argument("--out", type=Path, default=Path(r"D:\tmp\unlabeled-10k-real"), help="Output directory.")
    ap.add_argument("--workers", type=int, default=8, help="Parallel worker threads (6-8 recommended).")
    ap.add_argument("--max-dim", type=int, default=1600, help="Max width/height in pixels.")
    ap.add_argument("--max-pages-per-item", type=int, default=40, help="Max pages sampled per PDF.")
    ap.add_argument("--max-pdf-mb", type=int, default=35, help="Skip PDFs larger than this (MB).")
    ap.add_argument("--wc-quota", type=int, default=250, help="Max images from Wikimedia Commons.")
    ap.add_argument("--ia-sheetmusic-quota", type=int, default=2200, help="Max images from pre-1928 IA sheet music.")
    ap.add_argument("--resume", action="store_true", default=True, help="Resume existing harvest (default: on).")
    ap.add_argument("--no-resume", dest="resume", action="store_false", help="Start fresh.")
    ap.add_argument(
        "--cleanup-only",
        action="store_true",
        default=False,
        help="Run CPU RowNet v2 cleanup pass over existing images and exit.",
    )
    args = ap.parse_args(argv)

    if args.cleanup_only:
        summary = cleanup_existing_with_clip(out_dir=args.out, workers=args.workers)
        return 0 if summary["leaked_eval"] == 0 else 2

    summary = run_harvest(
        out_dir=args.out,
        target=args.target,
        workers=args.workers,
        max_dim=args.max_dim,
        max_pages_per_item=args.max_pages_per_item,
        max_pdf_mb=args.max_pdf_mb,
        resume=args.resume,
        wc_quota=args.wc_quota,
        ia_sheetmusic_quota=args.ia_sheetmusic_quota,
    )
    return 0 if summary["leaked_eval"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
