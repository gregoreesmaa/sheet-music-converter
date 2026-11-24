#!/usr/bin/env python3
"""Value trial harness for Qwen2.5-VL-3B-Instruct (Apache-2.0) on OMR -> bekern.

Empirically measures ROI on an RTX 4080 SUPER (16 GB VRAM) in two modes:
  - ``--mode zero_shot``: Runs ``Qwen/Qwen2.5-VL-3B-Instruct`` in BF16 on a
    balanced slice of ``suite31_hw`` strips (e.g., 5 printed + 5 handwritten)
    to measure zero-shot music literacy, exact latency/strip (ms), and
    bar-math validity.
  - ``--mode lora_sft``: Fine-tunes a lightweight BF16 LoRA (r=16, alpha=32,
    all attention + MLP linear projections, gradient checkpointing enabled,
    ``max_pixels=512*28*28``) for ``--steps 300`` on a balanced mix of
    printed + synthetic/real HW training strips (strictly excluding ALL
    ``suite31_hw`` eval pieces via a hard overlap guard), then evaluates on a
    held-out printed + HW strip probe (Strict-F1, bar-math validity, ms/strip)
    and writes a JSON summary report to ``--out-json``.
  - ``--dry-run``: Fast CPU smoke test verifying dataset loading, eval-overlap
    guard, prompt formatting, bekern post-processing, and metric computation
    without downloading or loading the 3B model.

All HuggingFace caches are pinned to ``D:\\tmp\\hf-cache`` so ``C:`` stays clean.
"""

from __future__ import annotations

import os

# Pin HuggingFace cache to D: before importing transformers / huggingface_hub.
os.environ.setdefault("HF_HOME", r"D:\tmp\hf-cache")
os.environ.setdefault("HF_HUB_CACHE", r"D:\tmp\hf-cache\hub")

import argparse
from dataclasses import dataclass, field
import difflib
import json
from pathlib import Path
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image

_MVD_DIR = Path(__file__).resolve().parent
REPO_ROOT = _MVD_DIR.parent.parent
if str(_MVD_DIR) not in sys.path:
    sys.path.insert(0, str(_MVD_DIR))

import data as mvd_data  # noqa: E402
from unsup_bar_math import score_bekern_bar_math  # noqa: E402

try:
    from eval_sys_gate import canonicalize_bekern_tail_order  # noqa: E402
except Exception:  # pragma: no cover - fallback if eval_sys_gate unavailable
    def canonicalize_bekern_tail_order(toks: Sequence[str]) -> List[str]:
        return list(toks)


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_MAX_PIXELS = 512 * 28 * 28  # 401,408 px (~512 vision tokens)
DEFAULT_MIN_PIXELS = 64 * 28 * 28
DEFAULT_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

SUITE31_DIR_DEFAULT = REPO_ROOT / "data-gen" / "score-corpus" / "suite31"
SUITE31_HW_MANIFEST_DEFAULT = SUITE31_DIR_DEFAULT / "suite31_hw_manifest.json"
DEFAULT_PRINTED_DIRS = (
    REPO_ROOT / "tmp-train" / "mvd-sysdata-t32",
    REPO_ROOT / "tmp-train" / "mvd-sysdata",
    Path(r"D:\tmp\mvd-sysdata-v2"),
    Path(r"D:\tmp\leland-sysdata-pilot"),
)
DEFAULT_HW_JSONL = Path(r"D:\tmp\real-hw-corpus\all_labeled_hw.jsonl")
DEFAULT_HW_SYNTH_DIRS = (
    REPO_ROOT / "tmp-train" / "hw-engine" / "mvp_samples",
    Path(r"D:\tmp\real-hw-train"),
)

ASER_EXE = (
    REPO_ROOT
    / "converter"
    / "assembly-serialize"
    / "target"
    / "release"
    / "assembly-serialize.exe"
)
EHARNESS_EXE = (
    REPO_ROOT
    / "converter"
    / "eval-harness"
    / "target"
    / "release"
    / "eval-harness.exe"
)

VOCAB_SET: Set[str] = set(mvd_data.VOCAB)

SYSTEM_PROMPT = (
    "You are an expert Optical Music Recognition (OMR) transcriber. "
    "Given a cropped single-system sheet music strip, output ONLY the exact "
    "space-separated bekern token sequence for the system. "
    "Use canonical bekern tokens: clef-* (e.g. clef-G-2, clef-F-4), "
    "key+N/key-N (e.g. key+0, key-1), time-N-D (e.g. time-4-4), voice-N, "
    "durations (whole, half, quarter, eighth, 16th, 32nd, 64th, breve), "
    "pitch step (A-G) + octave (0-8) + optional alter (a-2, a-1, a1, a2), "
    "dot, rest <dur>, bar-rest, chord, grace, tie-start/tie-stop, "
    "slur-start/slur-stop, and bar. Output no prose, markdown, or explanation."
)

USER_PROMPT = (
    "Transcribe this sheet music system strip into a single line of "
    "space-separated bekern tokens."
)


@dataclass
class StripSample:
    """One system strip sample for training or evaluation."""

    sample_id: str
    piece_slug: str
    domain: str  # "printed" | "handwritten"
    image_path: str
    bekern_tokens: List[str] = field(default_factory=list)
    crop_box: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    source: str = ""

    def load_image(self) -> Image.Image:
        im = Image.open(self.image_path).convert("RGB")
        if self.crop_box is not None:
            x1, y1, x2, y2 = self.crop_box
            w, h = im.size
            x1 = max(0, min(w - 1, int(x1)))
            y1 = max(0, min(h - 1, int(y1)))
            x2 = max(x1 + 1, min(w, int(x2)))
            y2 = max(y1 + 1, min(h, int(y2)))
            im = im.crop((x1, y1, x2, y2))
        return im


# ---------------------------------------------------------------------------
# Eval-overlap guard (HARD: zero suite31_hw pieces may enter training!)
# ---------------------------------------------------------------------------

def _normalize_id(text: str) -> str:
    """Lowercase alphanumeric-only slug for robust cross-corpus matching."""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def build_suite31_blocklist(
    manifest_paths: Sequence[Path] = (SUITE31_HW_MANIFEST_DEFAULT,),
) -> Set[str]:
    """Collect normalized piece slugs & file stems from suite31_hw manifests."""
    blocked: Set[str] = set()
    for mp in manifest_paths:
        p = Path(mp)
        if not p.exists():
            continue
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        for piece in doc.get("pieces", []):
            slug = str(piece.get("slug", "")).strip()
            if slug:
                blocked.add(_normalize_id(slug))
                # Also strip leading "hw-" prefix so the underlying work slug
                # (e.g. "beethoven-op135", "schumann-...") is equally blocked.
                if slug.lower().startswith("hw-"):
                    blocked.add(_normalize_id(slug[3:]))
            for k in ("clean", "photo", "gt"):
                sub = piece.get(k)
                if isinstance(sub, dict) and sub.get("file"):
                    stem = Path(str(sub["file"])).stem
                    blocked.add(_normalize_id(stem))
                    if stem.lower().startswith("gt-p1-"):
                        blocked.add(_normalize_id(stem[6:]))
    # Drop overly short tokens (<5 chars) that could cause false positives.
    return {b for b in blocked if len(b) >= 5}


def sample_matches_blocklist(sample: StripSample, blocklist: Set[str]) -> Optional[str]:
    """Return the matched blocklist key if ``sample`` overlaps suite31_hw, else None."""
    if not blocklist:
        return None
    candidates = [
        _normalize_id(sample.piece_slug),
        _normalize_id(sample.sample_id),
        _normalize_id(Path(sample.image_path).stem),
    ]
    for cand in candidates:
        if not cand:
            continue
        if cand in blocklist:
            return cand
        for b in blocklist:
            if b in cand or (len(cand) >= 6 and cand in b):
                return b
    return None


def filter_training_samples(
    samples: Sequence[StripSample],
    blocklist: Set[str],
) -> Tuple[List[StripSample], List[str]]:
    """Filter out any sample matching ``blocklist``. Returns ``(kept, dropped_ids)``."""
    kept: List[StripSample] = []
    dropped: List[str] = []
    for s in samples:
        if sample_matches_blocklist(s, blocklist) is not None:
            dropped.append(s.sample_id)
        else:
            kept.append(s)
    return kept, dropped


def assert_no_eval_overlap(
    train_samples: Sequence[StripSample],
    blocklist: Set[str],
) -> None:
    """Hard invariant check: raise ValueError if any training sample hits suite31_hw."""
    leaks: List[Tuple[str, str]] = []
    for s in train_samples:
        hit = sample_matches_blocklist(s, blocklist)
        if hit is not None:
            leaks.append((s.sample_id, hit))
    if leaks:
        preview = ", ".join(f"{sid} (matched {hit})" for sid, hit in leaks[:5])
        raise ValueError(
            f"Eval-overlap guard tripped: {len(leaks)} training sample(s) overlap "
            f"suite31_hw: {preview}"
        )


# ---------------------------------------------------------------------------
# Prompt formatting & bekern post-processing
# ---------------------------------------------------------------------------

def format_omr_messages(
    image: Any,
    target_bekern: Optional[str] = None,
    system_prompt: str = SYSTEM_PROMPT,
    user_prompt: str = USER_PROMPT,
) -> List[Dict[str, Any]]:
    """Build Qwen2.5-VL chat messages for inference or supervised LoRA SFT."""
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]
    if target_bekern is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": target_bekern.strip()}],
            }
        )
    return messages


_FENCE_RE = re.compile(r"```[a-zA-Z0-9_-]*\s*([\s\S]*?)```", re.MULTILINE)


def extract_bekern_from_response(
    raw_text: str,
    *,
    strict_vocab: bool = True,
    canonicalize: bool = True,
) -> List[str]:
    """Post-process raw VLM output into a clean canonical bekern token list.

    Strips markdown code fences, normalizes whitespace, optionally filters to
    known ``data.VOCAB`` tokens, and applies ``canonicalize_bekern_tail_order``.
    """
    if not raw_text:
        return []
    text = raw_text.strip()
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    # Replace punctuation/newlines that VLMs sometimes inject around tokens.
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    raw_toks = [t.strip(" ,;\"'`") for t in text.split() if t.strip(" ,;\"'`")]
    if strict_vocab:
        toks = [t for t in raw_toks if t in VOCAB_SET]
    else:
        toks = raw_toks
    if canonicalize and toks:
        toks = canonicalize_bekern_tail_order(toks)
    return toks


# ---------------------------------------------------------------------------
# Metrics: Strict-F1 (native harness if available + exact token-F1 fallback)
# ---------------------------------------------------------------------------

def token_sequence_f1(ref_toks: Sequence[str], hyp_toks: Sequence[str]) -> float:
    """Sequence-aligned token F1 (difflib LCS) for fast offline / fallback scoring."""
    ref_list, hyp_list = list(ref_toks), list(hyp_toks)
    if not ref_list and not hyp_list:
        return 1.0
    if not ref_list or not hyp_list:
        return 0.0
    sm = difflib.SequenceMatcher(a=ref_list, b=hyp_list, autojunk=False)
    matches = sum(block.size for block in sm.get_matching_blocks())
    prec = matches / len(hyp_list)
    rec = matches / len(ref_list)
    return (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0.0 else 0.0


def score_bekern_strict_f1(
    ref_toks: Sequence[str],
    hyp_toks: Sequence[str],
    *,
    use_rust_harness: bool = True,
) -> Dict[str, Any]:
    """Compute Strict-F1 (via Rust ``assembly-serialize`` + ``eval-harness``
    when available, otherwise sequence-aligned token-F1) plus bar-math metrics.
    """
    bm = score_bekern_bar_math(hyp_toks)
    tok_f1 = token_sequence_f1(ref_toks, hyp_toks) if ref_toks else None

    strict_f1: Optional[float] = None
    scorer_backend = "token_f1_fallback"

    if (
        use_rust_harness
        and ref_toks
        and hyp_toks
        and ASER_EXE.exists()
        and EHARNESS_EXE.exists()
    ):
        with tempfile.TemporaryDirectory(prefix="qwen_vl_f1_") as td:
            td_p = Path(td)
            ref_txt = td_p / "ref-bekern.txt"
            hyp_txt = td_p / "hyp-bekern.txt"
            ref_txt.write_text(" ".join(ref_toks) + "\n", encoding="utf-8")
            hyp_txt.write_text(" ".join(hyp_toks) + "\n", encoding="utf-8")

            ref_ok = subprocess.run(
                [str(ASER_EXE), "bekern-decode", "--input", str(ref_txt), "--out", str(td_p), "--stem", "ref"],
                capture_output=True,
                text=True,
                timeout=30,
            ).returncode == 0
            hyp_ok = subprocess.run(
                [str(ASER_EXE), "bekern-decode", "--input", str(hyp_txt), "--out", str(td_p), "--stem", "hyp"],
                capture_output=True,
                text=True,
                timeout=30,
            ).returncode == 0

            ref_json = td_p / "ref-notes.json"
            hyp_json = td_p / "hyp-notes.json"
            if ref_ok and hyp_ok and ref_json.exists() and hyp_json.exists():
                rep = td_p / "f1.json"
                p = subprocess.run(
                    [
                        str(EHARNESS_EXE),
                        "strict-f1",
                        "--reference",
                        str(ref_json),
                        "--hypothesis",
                        str(hyp_json),
                        "--report",
                        str(rep),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if p.returncode == 0 and rep.exists():
                    try:
                        strict_f1 = float(json.loads(rep.read_text(encoding="utf-8"))["f1"])
                        scorer_backend = "eval-harness"
                    except Exception:
                        strict_f1 = 0.0
                        scorer_backend = "eval-harness-err"
            elif ref_ok and not hyp_ok:
                strict_f1 = 0.0
                scorer_backend = "bekern-decode-fail"

    if strict_f1 is None and tok_f1 is not None:
        strict_f1 = tok_f1

    return {
        "strict_f1": strict_f1,
        "token_f1": tok_f1,
        "scorer_backend": scorer_backend,
        "valid_bar_ratio": float(bm.valid_bar_ratio),
        "n_bars": int(bm.n_bars),
        "n_valid_bars": int(bm.n_valid_bars),
        "mean_beat_err": float(bm.mean_beat_err),
        "is_degenerate": bool(bm.is_degenerate),
        "degenerate_reason": bm.degenerate_reason,
        "n_hyp_tokens": len(hyp_toks),
        "n_ref_tokens": len(ref_toks),
    }


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_suite31_hw_eval_slice(
    suite31_dir: Path = SUITE31_DIR_DEFAULT,
    manifest_path: Path = SUITE31_HW_MANIFEST_DEFAULT,
    n_printed: int = 5,
    n_hw: int = 5,
    prefer_photo: bool = False,
) -> List[StripSample]:
    """Load a balanced slice of ``suite31_hw`` strips (printed + handwritten)."""
    if not manifest_path.exists():
        return []
    doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    printed_samples: List[StripSample] = []
    hw_samples: List[StripSample] = []

    for piece in doc.get("pieces", []):
        slug = str(piece.get("slug", "")).strip()
        is_hw = bool(piece.get("handwritten", False)) or slug.startswith("hw-")
        img_key = "photo" if (prefer_photo and "photo" in piece) else "clean"
        if img_key not in piece or not isinstance(piece[img_key], dict):
            continue
        img_path = suite31_dir / str(piece[img_key]["file"])
        if not img_path.exists():
            continue

        # Prefer the first manifest system_box if available; otherwise top strip
        # heuristic (first ~28% of page height) so VLM sees a single system.
        crop_box: Optional[Tuple[int, int, int, int]] = None
        sys_boxes = piece.get("system_boxes")
        if isinstance(sys_boxes, list) and sys_boxes:
            b = sys_boxes[0]
            if isinstance(b, (list, tuple)) and len(b) == 4:
                crop_box = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
        else:
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
                crop_box = (0, int(h * 0.08), w, int(h * 0.36))
            except Exception:
                crop_box = None

        sample = StripSample(
            sample_id=f"suite31_{slug}_s0",
            piece_slug=slug,
            domain="handwritten" if is_hw else "printed",
            image_path=str(img_path),
            bekern_tokens=[],
            crop_box=crop_box,
            source="suite31_hw",
        )
        if is_hw:
            if len(hw_samples) < n_hw:
                hw_samples.append(sample)
        else:
            if len(printed_samples) < n_printed:
                printed_samples.append(sample)
        if len(printed_samples) >= n_printed and len(hw_samples) >= n_hw:
            break

    return printed_samples + hw_samples


def discover_sysdata_strips(
    dirs: Sequence[Path],
    *,
    domain: str = "printed",
    max_samples: int = 2000,
    seed: int = 42,
) -> List[StripSample]:
    """Discover ``<stem>.png`` / ``<stem>_h0.png`` + ``<stem>-bekern.txt`` pairs."""
    found: List[StripSample] = []
    for d in dirs:
        p = Path(d)
        if not p.exists():
            continue
        for bk in sorted(p.glob("*-bekern.txt")):
            stem = bk.name[: -len("-bekern.txt")]
            # Prefer handwritten-ink variant if domain == "handwritten", else clean/degraded.
            if domain == "handwritten":
                cand_imgs = [
                    p / f"{stem}_h0.png",
                    p / f"{stem}_h0.jpg",
                    p / f"{stem}.png",
                ]
            else:
                cand_imgs = [
                    p / f"{stem}_d0.jpg",
                    p / f"{stem}.png",
                    p / f"{stem}.jpg",
                ]
            img_path = next((c for c in cand_imgs if c.exists()), None)
            if img_path is None:
                continue
            try:
                toks = bk.read_text(encoding="utf-8").split()
            except Exception:
                continue
            if len(toks) < 6:
                continue
            piece_slug = mvd_data.piece_key(stem)
            found.append(
                StripSample(
                    sample_id=f"{p.name}:{stem}",
                    piece_slug=piece_slug,
                    domain=domain,
                    image_path=str(img_path),
                    bekern_tokens=toks,
                    source=str(p),
                )
            )
            if len(found) >= max_samples * 3:
                break
        if len(found) >= max_samples * 3:
            break

    rng = random.Random(seed)
    rng.shuffle(found)
    return found[:max_samples]


def discover_hw_jsonl_samples(
    jsonl_path: Path = DEFAULT_HW_JSONL,
    *,
    max_samples: int = 200,
    max_tokens: int = 384,
) -> List[StripSample]:
    """Load real labeled HW samples from ``all_labeled_hw.jsonl`` (if present)."""
    p = Path(jsonl_path)
    if not p.exists():
        return []
    root = p.parent
    out: List[StripSample] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if not rec.get("bekern_ok") or not rec.get("bekern_path"):
            continue
        bk_path = root / str(rec["bekern_path"])
        img_paths = rec.get("image_paths") or []
        if not bk_path.exists() or not img_paths:
            continue
        first_img = root / str(img_paths[0])
        if not first_img.exists():
            continue
        try:
            toks = bk_path.read_text(encoding="utf-8").split()[:max_tokens]
        except Exception:
            continue
        if len(toks) < 6:
            continue
        wid = str(rec.get("id", first_img.stem))
        out.append(
            StripSample(
                sample_id=f"real_hw:{wid}",
                piece_slug=wid,
                domain="handwritten",
                image_path=str(first_img),
                bekern_tokens=toks,
                source=str(p),
            )
        )
        if len(out) >= max_samples:
            break
    return out


def make_synthetic_dry_run_samples(
    tmp_dir: Path,
    n_printed: int = 4,
    n_hw: int = 4,
) -> Tuple[List[StripSample], List[StripSample]]:
    """Create tiny valid in-temp PNG strips + bekern GT for CPU ``--dry-run``."""
    valid_bekerns = [
        "clef-G-2 key+0 time-4-4 voice-1 quarter C 4 quarter D 4 quarter E 4 quarter F 4 bar voice-1 half G 4 half A 4 bar",
        "clef-G-2 key+1 time-3-4 voice-1 quarter G 4 quarter A 4 quarter B 4 bar voice-1 half C 5 quarter B 4 bar",
        "clef-F-4 key-1 time-4-4 voice-1 half F 3 half G 3 bar voice-1 quarter A 3 quarter B 3 a-1 half C 4 bar",
        "clef-G-2 key+0 time-2-4 voice-1 eighth C 4 eighth D 4 quarter E 4 bar voice-1 half G 4 bar",
    ]
    train_samples: List[StripSample] = []
    eval_samples: List[StripSample] = []

    for split_name, bucket, np_cnt, nh_cnt in (
        ("train", train_samples, n_printed, n_hw),
        ("eval", eval_samples, max(2, n_printed // 2), max(2, n_hw // 2)),
    ):
        for dom, count in (("printed", np_cnt), ("handwritten", nh_cnt)):
            for i in range(count):
                slug = f"dryrun-{split_name}-{dom}-{i}"
                img_path = tmp_dir / f"{slug}.png"
                arr = np.full((96, 384, 3), 245 if dom == "printed" else 232, dtype=np.uint8)
                # Draw 5 horizontal staff lines.
                for r in range(28, 69, 10):
                    arr[r : r + 2, 16:368, :] = 30
                Image.fromarray(arr).save(img_path)
                toks = valid_bekerns[i % len(valid_bekerns)].split()
                bucket.append(
                    StripSample(
                        sample_id=f"{slug}-s0",
                        piece_slug=slug,
                        domain=dom,
                        image_path=str(img_path),
                        bekern_tokens=toks,
                        source="dry_run_synth",
                    )
                )
    return train_samples, eval_samples


# ---------------------------------------------------------------------------
# Model loading, inference, and BF16 LoRA SFT
# ---------------------------------------------------------------------------

def load_qwen_vl_model_and_processor(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    use_bf16: bool = True,
):
    """Load Qwen2.5-VL-3B-Instruct and its AutoProcessor in BF16."""
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if (use_bf16 and torch.cuda.is_available()) else torch.float32
    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    return model, processor


def attach_lora_adapters(
    model,
    *,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Sequence[str] = DEFAULT_LORA_TARGETS,
):
    """Wrap Qwen2.5-VL with lightweight BF16 LoRA + gradient checkpointing."""
    from peft import LoraConfig, get_peft_model

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    cfg = LoraConfig(
        r=int(r),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        target_modules=list(target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, cfg)
    return peft_model


def infer_single_strip(
    model,
    processor,
    sample: StripSample,
    *,
    max_new_tokens: int = 256,
    dry_run: bool = False,
) -> Tuple[str, List[str], float]:
    """Run inference on one strip; returns ``(raw_text, bekern_toks, latency_ms)``."""
    t0 = time.perf_counter()
    im = sample.load_image()
    messages = format_omr_messages(im, target_bekern=None)

    if dry_run:
        # Simulate realistic VLM response (including markdown fence to test post-proc).
        toks = sample.bekern_tokens or (
            "clef-G-2 key+0 time-4-4 voice-1 quarter C 4 quarter D 4 quarter E 4 quarter F 4 bar".split()
        )
        raw_text = f"```bekern\n{' '.join(toks)}\n```"
        parsed = extract_bekern_from_response(raw_text)
        ms = (time.perf_counter() - t0) * 1000.0
        return raw_text, parsed, ms

    import torch
    from qwen_vl_utils import process_vision_info

    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    dev = next(model.parameters()).device
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
        )
    prompt_len = int(inputs["input_ids"].shape[1])
    new_ids = gen_ids[:, prompt_len:]
    raw_text = processor.batch_decode(
        new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    parsed = extract_bekern_from_response(raw_text)
    ms = (time.perf_counter() - t0) * 1000.0
    return raw_text, parsed, ms


def run_lora_sft_loop(
    model,
    processor,
    train_samples: Sequence[StripSample],
    *,
    steps: int = 300,
    lr: float = 2e-4,
    grad_accum: int = 4,
    seed: int = 42,
    log_every: int = 25,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run BF16 LoRA SFT over balanced printed + HW strip samples."""
    if not train_samples:
        raise ValueError("train_samples is empty; cannot run LoRA SFT")

    rng = random.Random(seed)
    printed = [s for s in train_samples if s.domain == "printed"]
    hw = [s for s in train_samples if s.domain == "handwritten"]

    def _pick_sample() -> StripSample:
        if printed and hw:
            pool = printed if rng.random() < 0.5 else hw
            return rng.choice(pool)
        return rng.choice(train_samples)

    if dry_run:
        # Exercise message formatting and image loading on a handful of steps.
        n_check = min(steps, 6)
        for _ in range(n_check):
            s = _pick_sample()
            im = s.load_image()
            _ = format_omr_messages(im, target_bekern=" ".join(s.bekern_tokens))
        return {
            "steps_completed": n_check,
            "mean_train_loss": 0.0,
            "final_train_loss": 0.0,
            "train_secs": 0.0,
            "dry_run": True,
        }

    import torch
    from qwen_vl_utils import process_vision_info

    model.train()
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(lr),
        weight_decay=0.01,
    )
    dev = next(model.parameters()).device
    losses: List[float] = []
    t_start = time.perf_counter()
    opt.zero_grad(set_to_none=True)

    for step in range(1, int(steps) + 1):
        step_loss = 0.0
        for _ in range(max(1, int(grad_accum))):
            s = _pick_sample()
            im = s.load_image()
            target_str = " ".join(s.bekern_tokens)
            full_msgs = format_omr_messages(im, target_bekern=target_str)
            prompt_msgs = format_omr_messages(im, target_bekern=None)

            full_text = processor.apply_chat_template(
                full_msgs, tokenize=False, add_generation_prompt=False
            )
            prompt_text = processor.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(full_msgs)
            full_batch = processor(
                text=[full_text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            prompt_batch = processor(
                text=[prompt_text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            prompt_len = int(prompt_batch["input_ids"].shape[1])

            full_batch = {k: v.to(dev) for k, v in full_batch.items()}
            labels = full_batch["input_ids"].clone()
            labels[:, :prompt_len] = -100

            out = model(**full_batch, labels=labels)
            loss = out.loss / float(max(1, int(grad_accum)))
            loss.backward()
            step_loss += float(loss.detach().item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(step_loss)

        if step == 1 or step % max(1, int(log_every)) == 0 or step == int(steps):
            print(
                f"[qwen-vl lora_sft] step={step}/{steps} loss={step_loss:.4f} "
                f"elapsed={time.perf_counter() - t_start:.1f}s",
                flush=True,
            )

    model.eval()
    return {
        "steps_completed": int(steps),
        "mean_train_loss": float(statistics.mean(losses)) if losses else 0.0,
        "final_train_loss": float(losses[-1]) if losses else 0.0,
        "train_secs": round(time.perf_counter() - t_start, 2),
        "dry_run": False,
    }


def evaluate_probe(
    model,
    processor,
    eval_samples: Sequence[StripSample],
    *,
    max_new_tokens: int = 256,
    dry_run: bool = False,
    use_rust_harness: bool = True,
) -> Dict[str, Any]:
    """Evaluate model on a list of strips and aggregate per-domain metrics."""
    rows: List[Dict[str, Any]] = []
    for s in eval_samples:
        raw_text, hyp_toks, ms = infer_single_strip(
            model,
            processor,
            s,
            max_new_tokens=max_new_tokens,
            dry_run=dry_run,
        )
        metrics = score_bekern_strict_f1(
            s.bekern_tokens,
            hyp_toks,
            use_rust_harness=(use_rust_harness and not dry_run),
        )
        rows.append(
            {
                "sample_id": s.sample_id,
                "piece_slug": s.piece_slug,
                "domain": s.domain,
                "ms": round(ms, 2),
                **metrics,
                "raw_preview": raw_text[:160],
            }
        )

    def _agg(subset: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not subset:
            return {
                "n": 0,
                "mean_ms_per_strip": 0.0,
                "mean_valid_bar_ratio": 0.0,
                "degenerate_frac": 0.0,
                "mean_strict_f1": None,
            }
        f1s = [r["strict_f1"] for r in subset if r["strict_f1"] is not None]
        return {
            "n": len(subset),
            "mean_ms_per_strip": round(statistics.mean(r["ms"] for r in subset), 2),
            "mean_valid_bar_ratio": round(
                statistics.mean(r["valid_bar_ratio"] for r in subset), 4
            ),
            "degenerate_frac": round(
                statistics.mean(1.0 if r["is_degenerate"] else 0.0 for r in subset), 4
            ),
            "mean_strict_f1": round(statistics.mean(f1s), 4) if f1s else None,
        }

    return {
        "overall": _agg(rows),
        "printed": _agg([r for r in rows if r["domain"] == "printed"]),
        "handwritten": _agg([r for r in rows if r["domain"] == "handwritten"]),
        "per_strip": rows,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Qwen2.5-VL-3B-Instruct OMR value trial (zero_shot / lora_sft)."
    )
    ap.add_argument(
        "--mode",
        choices=("zero_shot", "lora_sft"),
        default="zero_shot",
        help="Trial mode: zero_shot on suite31_hw slice or lora_sft + held-out probe.",
    )
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--n-printed", type=int, default=5)
    ap.add_argument("--n-hw", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--printed-dirs",
        nargs="*",
        default=[str(p) for p in DEFAULT_PRINTED_DIRS],
        help="Directories containing printed <stem>.png + <stem>-bekern.txt strips.",
    )
    ap.add_argument(
        "--hw-jsonl",
        default=str(DEFAULT_HW_JSONL),
        help="Optional labeled real HW JSONL (all_labeled_hw.jsonl).",
    )
    ap.add_argument(
        "--hw-synth-dirs",
        nargs="*",
        default=[str(p) for p in DEFAULT_HW_SYNTH_DIRS],
        help="Optional synthetic HW strip directories.",
    )
    ap.add_argument(
        "--suite31-dir",
        default=str(SUITE31_DIR_DEFAULT),
    )
    ap.add_argument(
        "--suite31-manifest",
        default=str(SUITE31_HW_MANIFEST_DEFAULT),
    )
    ap.add_argument("--out-json", default=None, help="Path to write JSON summary report.")
    ap.add_argument("--out-dir", default=None, help="Optional dir to save LoRA adapter.")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Fast CPU smoke test without downloading or loading the 3B model.",
    )
    return ap


def run_trial(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute the configured Qwen2.5-VL trial and return the summary dict."""
    blocklist = build_suite31_blocklist([Path(args.suite31_manifest)])
    tmp_ctx: Optional[tempfile.TemporaryDirectory] = None

    try:
        if args.dry_run:
            tmp_ctx = tempfile.TemporaryDirectory(prefix="qwen_vl_dryrun_")
            train_pool, eval_probe = make_synthetic_dry_run_samples(
                Path(tmp_ctx.name),
                n_printed=max(2, args.n_printed),
                n_hw=max(2, args.n_hw),
            )
            model, processor = None, None
        else:
            model, processor = load_qwen_vl_model_and_processor(
                args.model_id,
                max_pixels=args.max_pixels,
            )
            train_pool, eval_probe = [], []

        if args.mode == "zero_shot":
            if not args.dry_run:
                eval_probe = load_suite31_hw_eval_slice(
                    suite31_dir=Path(args.suite31_dir),
                    manifest_path=Path(args.suite31_manifest),
                    n_printed=args.n_printed,
                    n_hw=args.n_hw,
                )
            eval_summary = evaluate_probe(
                model,
                processor,
                eval_probe,
                max_new_tokens=args.max_new_tokens,
                dry_run=args.dry_run,
            )
            report: Dict[str, Any] = {
                "trial": "qwen2_5_vl_3b_omr",
                "mode": "zero_shot",
                "model_id": args.model_id,
                "hf_home": os.environ.get("HF_HOME"),
                "dry_run": bool(args.dry_run),
                "eval": eval_summary,
            }
        else:
            # lora_sft mode: gather printed + HW training strips, enforce hard
            # suite31_hw overlap guard, split held-out probe by piece_slug.
            dropped_overlap: List[str] = []
            if not args.dry_run:
                printed_raw = discover_sysdata_strips(
                    [Path(d) for d in args.printed_dirs],
                    domain="printed",
                    max_samples=1500,
                    seed=args.seed,
                )
                hw_raw = discover_hw_jsonl_samples(Path(args.hw_jsonl), max_samples=300)
                hw_raw.extend(
                    discover_sysdata_strips(
                        [Path(d) for d in args.hw_synth_dirs],
                        domain="handwritten",
                        max_samples=500,
                        seed=args.seed + 1,
                    )
                )
                # Also harvest any _h0 hand-ink variants from printed dirs if HW pool is small.
                if len(hw_raw) < 50:
                    hw_raw.extend(
                        discover_sysdata_strips(
                            [Path(d) for d in args.printed_dirs],
                            domain="handwritten",
                            max_samples=500,
                            seed=args.seed + 2,
                        )
                    )
                all_raw = printed_raw + hw_raw
                clean_pool, dropped_overlap = filter_training_samples(all_raw, blocklist)

                # Piece-level split for held-out probe (printed + HW).
                by_dom: Dict[str, Dict[str, List[StripSample]]] = {
                    "printed": {},
                    "handwritten": {},
                }
                for s in clean_pool:
                    by_dom.setdefault(s.domain, {}).setdefault(s.piece_slug, []).append(s)

                train_pool = []
                eval_probe = []
                for dom, target_eval in (("printed", args.n_printed), ("handwritten", args.n_hw)):
                    slugs = sorted(by_dom.get(dom, {}).keys())
                    rng = random.Random(args.seed + (0 if dom == "printed" else 17))
                    rng.shuffle(slugs)
                    for sl in slugs:
                        grp = by_dom[dom][sl]
                        if len(eval_probe) < (args.n_printed + args.n_hw) and sum(
                            1 for e in eval_probe if e.domain == dom
                        ) < target_eval:
                            eval_probe.append(grp[0])
                        else:
                            train_pool.extend(grp)
            else:
                train_pool, dropped_overlap = filter_training_samples(train_pool, blocklist)

            # Hard invariant: ZERO suite31_hw pieces may ever enter LoRA SFT.
            assert_no_eval_overlap(train_pool, blocklist)

            if not args.dry_run:
                model = attach_lora_adapters(
                    model,
                    r=args.lora_r,
                    alpha=args.lora_alpha,
                )

            train_stats = run_lora_sft_loop(
                model,
                processor,
                train_pool,
                steps=args.steps,
                lr=args.lr,
                grad_accum=args.grad_accum,
                seed=args.seed,
                dry_run=args.dry_run,
            )

            if args.out_dir and not args.dry_run:
                out_dir = Path(args.out_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(out_dir))
                processor.save_pretrained(str(out_dir))

            eval_summary = evaluate_probe(
                model,
                processor,
                eval_probe,
                max_new_tokens=args.max_new_tokens,
                dry_run=args.dry_run,
            )
            report = {
                "trial": "qwen2_5_vl_3b_omr",
                "mode": "lora_sft",
                "model_id": args.model_id,
                "hf_home": os.environ.get("HF_HOME"),
                "dry_run": bool(args.dry_run),
                "lora_config": {
                    "r": args.lora_r,
                    "alpha": args.lora_alpha,
                    "target_modules": list(DEFAULT_LORA_TARGETS),
                    "max_pixels": args.max_pixels,
                },
                "data_guard": {
                    "suite31_blocklist_size": len(blocklist),
                    "dropped_overlap_count": len(dropped_overlap),
                    "train_samples": len(train_pool),
                    "eval_probe_samples": len(eval_probe),
                },
                "train": train_stats,
                "eval": eval_summary,
            }

        if args.out_json:
            out_p = Path(args.out_json)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

        return report
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)
    report = run_trial(args)
    overall = report["eval"]["overall"]
    print(
        f"[trial_qwen_vl_omr] mode={report['mode']} dry_run={report['dry_run']} "
        f"n={overall['n']} strict_f1={overall['mean_strict_f1']} "
        f"bar_math={overall['mean_valid_bar_ratio']:.4f} "
        f"ms/strip={overall['mean_ms_per_strip']:.2f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
