"""Build a 10,000-image completely unlabeled, permissive-only sheet-music corpus manifest.

Strictly CPU-only, zero-copy JSONL referencing verified readable images on disk
(plus deterministic non-overlapping horizontal band crops for real handwritten pages
saved under ``<out_dir>/hw_crops/``).

Strict licensing & eval-seal rules:
- ZERO ``corpus10x-NC`` / ``cvc-muscima`` / non-commercial material (``nc_count == 0``).
- ZERO overlap with ``gate20`` (``exclusion.json``, ``gate_manifest.json``, ``dev-12``,
  ``test-8``, ``held_out_4page_gate``, ``quarantine.txt``), ``suite31``, or the 5 held-out
  ``suite31_hw`` works (``beethoven-op135``, ``mozart-k464``, ``mozart-k387``,
  ``mozart-k421``, ``schubert-d810``) — verified programmatically (``leaked_eval == 0``).
- ZERO labels or target fields in the output JSONL.

Usage:
  .\\.venv\\Scripts\\python.exe tmp-train/mvd/build_unlabeled_10k.py
  .\\.venv\\Scripts\\python.exe tmp-train/mvd/build_unlabeled_10k.py --out-dir D:\\tmp\\unlabeled-10k --seed 20261123
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = Path(r"D:\tmp\unlabeled-10k")
DEFAULT_SEED = 20261123
DEFAULT_TOTAL = 10000

VALID_DOMAINS = ("photo", "clean_render", "hand_ink_scan", "real_handwritten")
VALID_GRANULARITIES = ("page", "strip")
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}

FORBIDDEN_LABEL_KEYS = frozenset({
    "label", "labels", "target", "targets", "bekern", "musicxml", "gt",
    "ground_truth", "transcription", "tokens", "notes", "measures",
})

REQUIRED_ROW_KEYS = frozenset({
    "id", "path", "domain", "granularity", "width", "height", "source_pool", "license",
})

# Canonical 5 held-out handwritten eval works in suite31_hw
DEFAULT_HELD_OUT_HW_WORKS = (
    "beethoven-op135",
    "mozart-k464",
    "mozart-k387",
    "mozart-k421",
    "schubert-d810",
)

# Forbidden non-commercial / license-declined path or filename tokens
FORBIDDEN_NC_TOKENS = (
    "corpus10x-nc",
    "cvc-muscima",
    "muscima",
    "dcml",
    "cc-by-nc",
    "cc_by_nc",
    "noncommercial",
    "non-commercial",
    "homus",
)

FORBIDDEN_EVAL_DIR_NAMES = frozenset({
    "gate20", "suite31", "dev-12", "test-8", "dev-probe", "test-probe",
})

TARGET_DOMAIN_QUOTAS = {
    "photo": 3500,
    "clean_render": 3000,
    "hand_ink_scan": 2500,
    "real_handwritten": 1000,
}


@dataclass
class ExclusionPolicy:
    """Programmatic eval-seal and NC-licensing filter."""

    gate20_ids: Set[str] = field(default_factory=set)
    gate20_scores: Set[str] = field(default_factory=set)
    gate20_upstream_ids: Set[str] = field(default_factory=set)
    suite31_slugs: Set[str] = field(default_factory=set)
    held_out_hw_works: Set[str] = field(default_factory=lambda: set(DEFAULT_HELD_OUT_HW_WORKS))
    nc_tokens: Tuple[str, ...] = FORBIDDEN_NC_TOKENS
    _upstream_re: Optional[re.Pattern] = field(default=None, init=False, repr=False)
    _hw_re: Optional[re.Pattern] = field(default=None, init=False, repr=False)
    _nc_re: Optional[re.Pattern] = field(default=None, init=False, repr=False)

    def _compile(self) -> None:
        if self.gate20_upstream_ids:
            alts = "|".join(re.escape(u) for u in sorted(self.gate20_upstream_ids))
            self._upstream_re = re.compile(rf"(?<![a-z0-9])(?:{alts})(?![0-9])")
        else:
            self._upstream_re = None
        if self.held_out_hw_works:
            alts = "|".join(re.escape(w) for w in sorted(self.held_out_hw_works))
            self._hw_re = re.compile(alts)
        else:
            self._hw_re = None
        if self.nc_tokens:
            alts = "|".join(re.escape(t) for t in self.nc_tokens)
            self._nc_re = re.compile(alts)
        else:
            self._nc_re = None

    @classmethod
    def from_repo(cls, repo_root: Path = REPO_ROOT) -> "ExclusionPolicy":
        policy = cls()
        g20_dir = repo_root / "data-gen" / "score-corpus" / "gate20"
        s31_dir = repo_root / "data-gen" / "score-corpus" / "suite31"

        excl_path = g20_dir / "exclusion.json"
        if excl_path.is_file():
            data = json.loads(excl_path.read_text(encoding="utf-8"))
            for key in ("do_not_train", "held_out_4page_gate"):
                for item in data.get(key, []):
                    s = str(item).strip().lower()
                    if s:
                        policy.gate20_ids.add(s)
                        # Parent score ID before page suffix, e.g. ds0072-3 -> ds0072
                        base = s.rsplit("-", 1)[0] if "-" in s else s
                        if base:
                            policy.gate20_scores.add(base)

        g20_manifest = g20_dir / "gate_manifest.json"
        if g20_manifest.is_file():
            data = json.loads(g20_manifest.read_text(encoding="utf-8"))
            for page in data.get("pages", []):
                pid = str(page.get("id", "")).strip().lower()
                if pid:
                    policy.gate20_ids.add(pid)
                score = str(page.get("score", "")).strip().lower()
                if score:
                    policy.gate20_scores.add(score)
                src = str(page.get("source", "")).lower()
                for m in re.findall(r"(?:lc|sq)\d+", src):
                    policy.gate20_upstream_ids.add(m)

        q_path = repo_root / "tmp-train" / "mvd-data" / "quarantine.txt"
        if q_path.is_file():
            for line in q_path.read_text(encoding="utf-8").splitlines():
                s = line.strip().lower()
                if s and not s.startswith("#"):
                    policy.gate20_ids.add(s)

        s31_manifest = s31_dir / "suite31_manifest.json"
        if s31_manifest.is_file():
            data = json.loads(s31_manifest.read_text(encoding="utf-8"))
            for piece in data.get("pieces", []):
                slug = str(piece.get("slug", "")).strip().lower()
                if slug:
                    policy.suite31_slugs.add(slug)

        s31_hw = s31_dir / "suite31_hw_only_manifest.json"
        if s31_hw.is_file():
            data = json.loads(s31_hw.read_text(encoding="utf-8"))
            for piece in data.get("pieces", []):
                slug = str(piece.get("slug", "")).strip().lower()
                if slug:
                    policy.suite31_slugs.add(slug)
                    if slug.startswith("hw-"):
                        policy.held_out_hw_works.add(slug[3:])

        policy._compile()
        return policy

    def is_forbidden_nc(self, path: Path | str) -> bool:
        if self._nc_re is None:
            self._compile()
        norm = str(path).replace("\\", "/").lower()
        return bool(self._nc_re and self._nc_re.search(norm))

    def is_leaked_eval(self, path: Path | str) -> bool:
        if self._hw_re is None and self.held_out_hw_works:
            self._compile()
        p = Path(path)
        norm = str(p).replace("\\", "/").lower()
        parts = [part.lower() for part in p.parts]

        # 1. Any eval directory in path
        if any(part in FORBIDDEN_EVAL_DIR_NAMES for part in parts):
            return True

        # 2. Any of the 5 held-out handwritten eval works anywhere in path
        if self._hw_re is not None and self._hw_re.search(norm):
            return True

        # 3. Upstream lc/sq IDs (fast single compiled regex)
        if self._upstream_re is not None and self._upstream_re.search(norm):
            return True

        stem = p.stem.lower()
        # Strip common degradation / variant suffixes to get base stem
        base_stem = re.sub(r"(_degraded|_hand|_d\d+|_h\d+|_marked|_plain)$", "", stem)

        # 4. Gate20 exact page IDs or score prefixes
        if stem in self.gate20_ids or base_stem in self.gate20_ids:
            return True
        if stem in self.gate20_scores or base_stem in self.gate20_scores:
            return True
        prefix_dash = stem.split("-", 1)[0]
        prefix_under = stem.split("_", 1)[0]
        if prefix_dash in self.gate20_scores or prefix_under in self.gate20_scores:
            return True
        if prefix_dash in self.gate20_ids or prefix_under in self.gate20_ids:
            return True

        # 5. Suite31 slugs (exact stem or slug prefix, avoiding false-positive on mozart-k458 vs mozart-k458-1)
        if stem in self.suite31_slugs or base_stem in self.suite31_slugs:
            return True
        if stem.startswith("gt-p1-") and stem[6:] in self.suite31_slugs:
            return True
        if prefix_under in self.suite31_slugs:
            return True
        for part in parts:
            if part in self.suite31_slugs or (part.startswith("cr-") and part[3:] in self.suite31_slugs):
                return True

        return False


def probe_image_size(path: Path) -> Optional[Tuple[int, int]]:
    """Read image (width, height) from header without full decode; None if corrupt."""
    try:
        with Image.open(path) as im:
            w, h = im.size
            if w > 0 and h > 0:
                return int(w), int(h)
    except Exception:
        return None
    return None


def _scan_flat_dir(
    directory: Path,
    *,
    predicate,
    policy: ExclusionPolicy,
    stats: Counter,
) -> List[Path]:
    """Fast os.scandir over a flat directory with strict policy filtering."""
    out: List[Path] = []
    if not directory.is_dir():
        return out
    for entry in os.scandir(directory):
        if not entry.is_file():
            continue
        name = entry.name
        ext = os.path.splitext(name)[1].lower()
        if ext not in IMG_EXTS:
            continue
        if not predicate(name):
            continue
        p = Path(entry.path)
        if policy.is_forbidden_nc(p):
            stats["rejected_nc"] += 1
            continue
        if policy.is_leaked_eval(p):
            stats["rejected_eval"] += 1
            continue
        out.append(p)
    out.sort(key=lambda x: str(x).lower())
    return out


def _scan_recursive_dir(
    directory: Path,
    *,
    predicate,
    policy: ExclusionPolicy,
    stats: Counter,
) -> List[Path]:
    """Recursive os.walk scan with strict policy filtering."""
    out: List[Path] = []
    if not directory.is_dir():
        return out
    for root, dirs, files in os.walk(directory):
        # Prune NC / eval directories immediately
        kept_dirs = []
        for d in sorted(dirs):
            dp = Path(root) / d
            if policy.is_forbidden_nc(dp):
                stats["rejected_nc"] += 1
                continue
            if policy.is_leaked_eval(dp):
                stats["rejected_eval"] += 1
                continue
            kept_dirs.append(d)
        dirs[:] = kept_dirs
        for name in sorted(files):
            ext = os.path.splitext(name)[1].lower()
            if ext not in IMG_EXTS:
                continue
            if not predicate(name):
                continue
            p = Path(root) / name
            if policy.is_forbidden_nc(p):
                stats["rejected_nc"] += 1
                continue
            if policy.is_leaked_eval(p):
                stats["rejected_eval"] += 1
                continue
            out.append(p)
    out.sort(key=lambda x: str(x).lower())
    return out


def _sample_from_candidates(
    candidates: List[Path],
    quota: int,
    *,
    domain: str,
    granularity: str,
    source_pool: str,
    license_str: str,
    rng: random.Random,
    seen_keys: Set[str],
    stats: Counter,
    dedup_by_name: bool = True,
) -> List[Dict]:
    """Deterministically sample up to `quota` valid, readable, non-duplicate images."""
    if quota <= 0 or not candidates:
        return []
    pool = list(candidates)
    rng.shuffle(pool)
    rows: List[Dict] = []
    for p in pool:
        if len(rows) >= quota:
            break
        key = p.name.lower() if dedup_by_name else str(p.resolve()).lower()
        if key in seen_keys:
            continue
        size = probe_image_size(p)
        if size is None:
            stats["corrupt_skipped"] += 1
            continue
        seen_keys.add(key)
        w, h = size
        rows.append({
            "path": str(p.resolve()),
            "domain": domain,
            "granularity": granularity,
            "width": w,
            "height": h,
            "source_pool": source_pool,
            "license": license_str,
        })
    return rows


def generate_hw_band_crops(
    hw_pages: Sequence[Path],
    target_crops: int,
    crops_dir: Path,
    *,
    rng: random.Random,
    policy: ExclusionPolicy,
    stats: Counter,
) -> List[Dict]:
    """Generate deterministic non-overlapping horizontal system/band crops from non-eval HW pages."""
    if target_crops <= 0 or not hw_pages:
        return []
    crops_dir.mkdir(parents=True, exist_ok=True)

    # Build deterministic non-overlapping band specs across all non-eval HW pages
    band_specs: List[Tuple[Path, str, int, int, int, int, int]] = []
    for page_path in hw_pages:
        if policy.is_forbidden_nc(page_path) or policy.is_leaked_eval(page_path):
            stats["rejected_eval"] += 1
            continue
        size = probe_image_size(page_path)
        if size is None:
            stats["corrupt_skipped"] += 1
            continue
        w, h = size
        if w < 128 or h < 128:
            continue
        work = page_path.parent.name
        x0 = int(round(0.03 * w))
        x1 = int(round(0.97 * w))
        y0 = int(round(0.06 * h))
        y1 = int(round(0.94 * h))
        n_bands = 3 if h >= 1400 else 2
        band_h = (y1 - y0) // n_bands
        if band_h < 64:
            n_bands = 2
            band_h = (y1 - y0) // n_bands
        for k in range(n_bands):
            by0 = y0 + k * band_h
            by1 = y0 + (k + 1) * band_h
            band_specs.append((page_path, work, k, x0, by0, x1, by1))

    # Deterministic shuffle & select
    band_specs.sort(key=lambda t: (t[1].lower(), t[0].name.lower(), t[2]))
    rng.shuffle(band_specs)
    selected = band_specs[:target_crops]

    # Group by source page so each source image is opened once
    by_page: Dict[Path, List[Tuple[str, int, int, int, int, int]]] = {}
    for page_path, work, k, x0, by0, x1, by1 in selected:
        by_page.setdefault(page_path, []).append((work, k, x0, by0, x1, by1))

    rows: List[Dict] = []
    for page_path in sorted(by_page.keys(), key=lambda x: str(x).lower()):
        specs = sorted(by_page[page_path], key=lambda t: t[1])
        try:
            with Image.open(page_path) as im:
                rgb = im.convert("RGB")
                for work, k, x0, by0, x1, by1 in specs:
                    out_name = f"{work}__{page_path.stem}__band{k}.jpg"
                    out_path = crops_dir / out_name
                    if not out_path.is_file():
                        crop = rgb.crop((x0, by0, x1, by1))
                        tmp_path = crops_dir / (out_name + ".tmp")
                        crop.save(tmp_path, format="JPEG", quality=95)
                        os.replace(tmp_path, out_path)
                    cw, ch = x1 - x0, by1 - by0
                    rows.append({
                        "path": str(out_path.resolve()),
                        "domain": "real_handwritten",
                        "granularity": "strip",
                        "width": int(cw),
                        "height": int(ch),
                        "source_pool": "handwritten500_band_crops",
                        "license": "Public Domain (IMSLP Holographs)",
                    })
        except Exception:
            stats["corrupt_skipped"] += 1
            continue

    # Preserve deterministic order matching selected order
    rows.sort(key=lambda r: r["path"].lower())
    return rows


def audit_manifest(
    rows: Sequence[Dict],
    policy: ExclusionPolicy,
    *,
    verify_headers: bool = True,
) -> Dict[str, int]:
    """Programmatically verify zero eval leakage, zero NC, zero corrupt images, and strict schema."""
    leaked_eval = 0
    nc_count = 0
    corrupt_count = 0
    label_key_violations = 0
    schema_violations = 0
    duplicate_paths = 0
    seen_paths: Set[str] = set()

    for row in rows:
        keys = set(row.keys())
        if not REQUIRED_ROW_KEYS.issubset(keys):
            schema_violations += 1
        if keys & FORBIDDEN_LABEL_KEYS:
            label_key_violations += 1
        if row.get("domain") not in VALID_DOMAINS:
            schema_violations += 1
        if row.get("granularity") not in VALID_GRANULARITIES:
            schema_violations += 1

        p_str = str(row.get("path", ""))
        norm_p = p_str.lower()
        if norm_p in seen_paths:
            duplicate_paths += 1
        seen_paths.add(norm_p)

        p = Path(p_str)
        if policy.is_forbidden_nc(p):
            nc_count += 1
        if policy.is_leaked_eval(p):
            leaked_eval += 1
        if verify_headers:
            sz = probe_image_size(p)
            if sz is None or sz[0] <= 0 or sz[1] <= 0:
                corrupt_count += 1

    return {
        "leaked_eval": leaked_eval,
        "nc_count": nc_count,
        "corrupt_count": corrupt_count,
        "label_key_violations": label_key_violations,
        "schema_violations": schema_violations,
        "duplicate_paths": duplicate_paths,
    }


def _is_degraded_jpg(name: str) -> bool:
    low = name.lower()
    return low.endswith(".jpg") and bool(re.search(r"_d\d+\.jpg$", low))


def _is_clean_png(name: str) -> bool:
    low = name.lower()
    return low.endswith(".png") and not re.search(r"(_d\d+|_h\d+|_hand|_degraded)\.png$", low)


def _is_hand_ink(name: str) -> bool:
    low = name.lower()
    return bool(re.search(r"_h\d+\.(png|jpg|jpeg)$", low))


def _is_hw_synth_page(name: str) -> bool:
    low = name.lower()
    if any(bad in low for bad in ("grid", "zoom", "canny", "control", "clean")):
        return False
    return low.endswith("_hand.png") or (low.startswith("hw_ds") and low.endswith(".png"))


def build_corpus(
    *,
    out_dir: Path = DEFAULT_OUT_DIR,
    repo_root: Path = REPO_ROOT,
    data_root: Path = Path(r"D:\tmp"),
    seed: int = DEFAULT_SEED,
    domain_quotas: Optional[Dict[str, int]] = None,
) -> Dict:
    """Curate and materialize the unlabeled 10k manifest and summary."""
    quotas = dict(TARGET_DOMAIN_QUOTAS if domain_quotas is None else domain_quotas)
    policy = ExclusionPolicy.from_repo(repo_root)
    rng = random.Random(seed)
    stats: Counter = Counter()
    seen_keys: Set[str] = set()
    all_rows: List[Dict] = []

    pdmx_root = data_root / "corpus10x" / "pdmx"
    corpus_root = repo_root / "data-gen" / "score-corpus"

    # ------------------------------------------------------------------
    # 1. PHOTO domain (target 3,500: simulated phone photos + degraded strips)
    # ------------------------------------------------------------------
    photo_target = quotas["photo"]
    photo_rows: List[Dict] = []

    photo_pools = [
        (
            "s16sim_photos",
            data_root / "s16sim-photos",
            False,
            lambda n: True,
            "page",
            "CC0-1.0 (OpenScore)",
            120,
        ),
        (
            "score_corpus_degraded10",
            corpus_root / "degraded10",
            False,
            lambda n: True,
            "page",
            "CC0-1.0 (OpenScore Lieder)",
            20,
        ),
        (
            "big500_degraded10",
            corpus_root / "big500-degraded10",
            False,
            lambda n: True,
            "page",
            "CC0-1.0 (OpenScore Lieder)",
            20,
        ),
        (
            "pdmx_leland_sysdata_photo",
            pdmx_root / "leland-sysdata",
            False,
            _is_degraded_jpg,
            "strip",
            "CC0-1.0 / Public Domain (PDMX)",
            1200,
        ),
        (
            "pdmx_quota_sysdata_photo",
            pdmx_root / "quota-sysdata",
            False,
            _is_degraded_jpg,
            "strip",
            "CC0-1.0 / Public Domain (PDMX)",
            1400,
        ),
        (
            "pdmx_unified_sysdata_photo",
            pdmx_root / "unified-sysdata",
            False,
            _is_degraded_jpg,
            "strip",
            "CC0-1.0 / Public Domain (PDMX)",
            photo_target,  # fill remaining to hit exact photo_target
        ),
    ]
    for pool_name, pool_dir, recursive, pred, gran, lic, cap in photo_pools:
        rem = photo_target - len(photo_rows)
        if rem <= 0:
            break
        cands = (
            _scan_recursive_dir(pool_dir, predicate=pred, policy=policy, stats=stats)
            if recursive
            else _scan_flat_dir(pool_dir, predicate=pred, policy=policy, stats=stats)
        )
        picked = _sample_from_candidates(
            cands,
            min(cap, rem),
            domain="photo",
            granularity=gran,
            source_pool=pool_name,
            license_str=lic,
            rng=rng,
            seen_keys=seen_keys,
            stats=stats,
        )
        photo_rows.extend(picked)

    # Fallback top-up from quota-sysdata if unified-sysdata had high dedup overlap
    if len(photo_rows) < photo_target:
        cands = _scan_flat_dir(
            pdmx_root / "quota-sysdata",
            predicate=_is_degraded_jpg,
            policy=policy,
            stats=stats,
        )
        photo_rows.extend(
            _sample_from_candidates(
                cands,
                photo_target - len(photo_rows),
                domain="photo",
                granularity="strip",
                source_pool="pdmx_quota_sysdata_photo",
                license_str="CC0-1.0 / Public Domain (PDMX)",
                rng=rng,
                seen_keys=seen_keys,
                stats=stats,
            )
        )
    all_rows.extend(photo_rows)

    # ------------------------------------------------------------------
    # 2. CLEAN_RENDER domain (target 3,000: multi-font pages + clean strips)
    # ------------------------------------------------------------------
    clean_target = quotas["clean_render"]
    clean_rows: List[Dict] = []

    clean_pools = [
        (
            "pdmx_quota_renders_leland",
            pdmx_root / "quota-renders-leland",
            False,
            _is_clean_png,
            "page",
            "CC0-1.0 / Public Domain (PDMX)",
            700,
        ),
        (
            "pdmx_quota_renders",
            pdmx_root / "quota-renders",
            False,
            _is_clean_png,
            "page",
            "CC0-1.0 / Public Domain (PDMX)",
            500,
        ),
        (
            "t32_renders_multifont",
            corpus_root / "t32-renders",
            True,
            _is_clean_png,
            "page",
            "CC0-1.0 (OpenScore)",
            500,
        ),
        (
            "big500_renders",
            corpus_root / "big500-renders",
            False,
            _is_clean_png,
            "page",
            "CC0-1.0 (OpenScore Lieder)",
            300,
        ),
        (
            "sq_renders",
            corpus_root / "sq-renders",
            False,
            _is_clean_png,
            "page",
            "CC0-1.0 (OpenScore StringQuartets)",
            200,
        ),
        (
            "openewld_renders",
            data_root / "corpus10x" / "openewld" / "renders",
            False,
            _is_clean_png,
            "page",
            "MIT / Public Domain (OpenEWLD)",
            150,
        ),
        (
            "pdmx_unified_sysdata_clean",
            pdmx_root / "unified-sysdata",
            False,
            _is_clean_png,
            "strip",
            "CC0-1.0 / Public Domain (PDMX)",
            400,
        ),
        (
            "pdmx_leland_sysdata_clean",
            pdmx_root / "leland-sysdata",
            False,
            _is_clean_png,
            "strip",
            "CC0-1.0 / Public Domain (PDMX)",
            clean_target,
        ),
    ]
    for pool_name, pool_dir, recursive, pred, gran, lic, cap in clean_pools:
        rem = clean_target - len(clean_rows)
        if rem <= 0:
            break
        cands = (
            _scan_recursive_dir(pool_dir, predicate=pred, policy=policy, stats=stats)
            if recursive
            else _scan_flat_dir(pool_dir, predicate=pred, policy=policy, stats=stats)
        )
        picked = _sample_from_candidates(
            cands,
            min(cap, rem),
            domain="clean_render",
            granularity=gran,
            source_pool=pool_name,
            license_str=lic,
            rng=rng,
            seen_keys=seen_keys,
            stats=stats,
        )
        clean_rows.extend(picked)
    all_rows.extend(clean_rows)

    # ------------------------------------------------------------------
    # 3. HAND_INK_SCAN domain (target 2,500: hand-ink / parchment variants)
    # ------------------------------------------------------------------
    hand_target = quotas["hand_ink_scan"]
    hand_rows: List[Dict] = []

    hw_synth_cands = _scan_recursive_dir(
        repo_root / "tmp-train" / "hw-synth",
        predicate=_is_hw_synth_page,
        policy=policy,
        stats=stats,
    )
    hand_rows.extend(
        _sample_from_candidates(
            hw_synth_cands,
            min(20, hand_target),
            domain="hand_ink_scan",
            granularity="page",
            source_pool="hw_synth_pages",
            license_str="CC0-1.0 (Synthetic Hand-Ink)",
            rng=rng,
            seen_keys=seen_keys,
            stats=stats,
        )
    )

    if len(hand_rows) < hand_target:
        h0_cands = _scan_flat_dir(
            pdmx_root / "quota-sysdata",
            predicate=_is_hand_ink,
            policy=policy,
            stats=stats,
        )
        hand_rows.extend(
            _sample_from_candidates(
                h0_cands,
                hand_target - len(hand_rows),
                domain="hand_ink_scan",
                granularity="strip",
                source_pool="pdmx_quota_sysdata_handink",
                license_str="CC0-1.0 / Public Domain (PDMX)",
                rng=rng,
                seen_keys=seen_keys,
                stats=stats,
            )
        )
    all_rows.extend(hand_rows)

    # ------------------------------------------------------------------
    # 4. REAL_HANDWRITTEN domain (target 1,000: non-eval pages + band crops)
    # ------------------------------------------------------------------
    hw_target = quotas["real_handwritten"]
    hw_root = corpus_root / "handwritten-500" / "handwritten"
    hw_page_cands = _scan_recursive_dir(
        hw_root,
        predicate=lambda n: True,
        policy=policy,
        stats=stats,
    )
    hw_page_rows = _sample_from_candidates(
        hw_page_cands,
        min(len(hw_page_cands), hw_target),
        domain="real_handwritten",
        granularity="page",
        source_pool="handwritten500_pages",
        license_str="Public Domain (IMSLP Holographs)",
        rng=rng,
        seen_keys=seen_keys,
        stats=stats,
        dedup_by_name=False,
    )
    hw_crop_rows = generate_hw_band_crops(
        hw_page_cands,
        hw_target - len(hw_page_rows),
        out_dir / "hw_crops",
        rng=rng,
        policy=policy,
        stats=stats,
    )
    all_rows.extend(hw_page_rows)
    all_rows.extend(hw_crop_rows)

    # Assign deterministic sequential IDs
    all_rows.sort(key=lambda r: (VALID_DOMAINS.index(r["domain"]), r["source_pool"], r["path"].lower()))
    for idx, row in enumerate(all_rows):
        row["id"] = f"u10k-{idx:05d}"

    # Programmatic audit before writing
    audit = audit_manifest(all_rows, policy, verify_headers=True)
    expected_total = sum(quotas.values())
    if len(all_rows) != expected_total:
        raise RuntimeError(f"Expected {expected_total} rows, got {len(all_rows)}")
    if audit["leaked_eval"] != 0:
        raise RuntimeError(f"EVAL LEAKAGE DETECTED: leaked_eval={audit['leaked_eval']}")
    if audit["nc_count"] != 0:
        raise RuntimeError(f"NC LICENSE VIOLATION: nc_count={audit['nc_count']}")
    if audit["corrupt_count"] != 0:
        raise RuntimeError(f"CORRUPT IMAGES DETECTED: corrupt_count={audit['corrupt_count']}")
    if audit["label_key_violations"] != 0:
        raise RuntimeError(f"LABEL KEYS DETECTED: {audit['label_key_violations']}")
    if audit["schema_violations"] != 0 or audit["duplicate_paths"] != 0:
        raise RuntimeError(f"SCHEMA/DUP VIOLATION: {audit}")

    # Atomic tmp + rename writes
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "unlabeled_10k_manifest.jsonl"
    manifest_tmp = out_dir / "unlabeled_10k_manifest.jsonl.tmp"
    h = hashlib.sha256()
    with open(manifest_tmp, "w", encoding="utf-8", newline="\n") as f:
        for row in all_rows:
            line = json.dumps(row, ensure_ascii=False) + "\n"
            f.write(line)
            h.update(line.encode("utf-8"))
    os.replace(manifest_tmp, manifest_path)

    domain_counts = Counter(r["domain"] for r in all_rows)
    gran_counts = Counter(r["granularity"] for r in all_rows)
    pool_counts = Counter(r["source_pool"] for r in all_rows)
    lic_counts = Counter(r["license"] for r in all_rows)

    summary = {
        "total": len(all_rows),
        "seed": seed,
        "domain_counts": {d: domain_counts[d] for d in VALID_DOMAINS},
        "granularity_counts": {g: gran_counts[g] for g in VALID_GRANULARITIES},
        "source_pool_counts": dict(sorted(pool_counts.items())),
        "license_counts": dict(sorted(lic_counts.items())),
        "leaked_eval": audit["leaked_eval"],
        "nc_count": audit["nc_count"],
        "corrupt_count": audit["corrupt_count"],
        "label_key_violations": audit["label_key_violations"],
        "duplicate_paths": audit["duplicate_paths"],
        "rejected_eval_candidates": int(stats["rejected_eval"]),
        "rejected_nc_candidates": int(stats["rejected_nc"]),
        "corrupt_skipped_during_scan": int(stats["corrupt_skipped"]),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": h.hexdigest(),
    }

    summary_path = out_dir / "summary.json"
    summary_tmp = out_dir / "summary.json.tmp"
    summary_tmp.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    os.replace(summary_tmp, summary_path)

    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build 10k unlabeled permissive sheet-music corpus manifest.")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    ap.add_argument("--data-root", type=Path, default=Path(r"D:\tmp"))
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args(argv)

    summary = build_corpus(
        out_dir=args.out_dir,
        repo_root=args.repo_root,
        data_root=args.data_root,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
