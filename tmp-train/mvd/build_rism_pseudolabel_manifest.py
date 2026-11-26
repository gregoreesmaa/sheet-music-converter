#!/usr/bin/env python3
"""Promoted ``ov-39`` RISM Pseudo-Label Manifest Builder for UniOMR v1 / v2.

Implements the ``PROMOTE``-verdict filtering & tiering pipeline from
``experiments/2026-11-26T01-38-ov-39-rism-pseudolabel-threshold-sweep/OUTCOME.md``:

  1. Loads any UniOMR checkpoint (auto-detecting ``v1`` ``UniOMRNet`` or ``v2``
     ``UniOMRv2Net``) with T4-safe BF16/FP32 precision selection.
  2. Decodes unlabeled archival RISM pages (``D:\\tmp\\unlabeled-10k-real\\rism_hw``
     or an extracted HF shard) using FSA-constrained MTP-4 self-speculative
     decoding at ``max_dim=1280``, ``gen_max_len=2048``.
  3. Computes exact per-token ``mean_logprob`` + label-free structural rewards
     (``beat_valid_frac`` = ``r_rhythm``, ``r_parse``, ``r_noloop``) using the
     exact ``ov-39`` formulas.
  4. Applies the promoted ``ov-39`` quality gate:
       - ``mean_logprob >= -0.35``
       - ``beat_valid_frac >= 0.70``
       - ``r_noloop >= 0.99`` (``r_noloop == 1.0``)
       - ``r_parse >= 0.5``
       - ``n_tokens >= 24``
     and tags ``beat_valid_frac >= 0.85`` as ``tier="high_rhythm"``
     (default ``sample_weight=1.75``; ``tier="standard"`` gets ``1.0``).
  5. Writes per-system ``*-bekern.txt`` files and an append-only resumable
     training-ready JSONL manifest matching ``uni_fullpage_iter11_clean.jsonl``
     (``image``, ``bekerns``, ``domain="real_hw_pseudo"``, plus quality/tier
     metadata).
  6. Supports ``--resume`` (append-only JSONL + scored ledger), deterministic
     ``--shard-idx / --num-shards`` parallelization across multiple GPUs/hosts,
     and ``--merge-shards`` to combine completed shard manifests.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from PIL import Image
import torch
import torch.nn.functional as F

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import eval_uni_suite31 as ev  # noqa: E402
import rlvr_grpo  # noqa: E402
import uni_omr  # noqa: E402
import uni_omr_v2  # noqa: E402


DEFAULT_INPUT_DIR = Path(r"D:\tmp\unlabeled-10k-real\rism_hw")
DEFAULT_OUT_MANIFEST = Path(r"D:\tmp\rism-pseudolabels-ov39\uni_rism_pseudolabels_ov39.jsonl")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp"}

# Keys exclusive to UniOMRv2Config / UniOMRv2Net state_dict used for auto-detection.
_V2_CFG_MARKERS = frozenset({
    "n_kv_heads",
    "n_axial_layers",
    "n_global_layers",
    "backbone_channels",
    "backbone_blocks",
    "use_staff_seg_aux",
    "use_stroke_hog_head",
    "irope_nope_every",
    "n_register_tokens",
})
_V2_SD_PREFIXES = (
    "deepstack_fusions.",
    "encoder.axial_layers.",
    "encoder.global_layers.",
    "backbone.p2_",
    "backbone.p3_",
    "staff_seg_head.",
    "stroke_hog_head.",
)


# ---------------------------------------------------------------------------
# Configuration & Scoring Helpers (Pure / CPU-Testable)
# ---------------------------------------------------------------------------

@dataclass
class Ov39GateConfig:
    """Promoted ``ov-39`` filtering thresholds and tiering parameters."""

    min_logprob: float = -0.35
    min_beat_valid: float = 0.70
    high_rhythm_beat_valid: float = 0.85
    min_r_parse: float = 0.50
    min_r_noloop: float = 0.99
    min_tokens: int = 24
    high_rhythm_weight: float = 1.75
    standard_weight: float = 1.0
    domain: str = "real_hw_pseudo"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def discover_images(input_dir: Path) -> List[Path]:
    """Discover all image files under ``input_dir`` in deterministic sorted order."""
    if input_dir.is_file() and input_dir.suffix.lower() in IMAGE_EXTS:
        return [input_dir]
    imgs = [
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    imgs.sort(key=lambda p: str(p).replace("\\", "/").lower())
    return imgs


def shard_images(
    images: Sequence[Path],
    shard_idx: int = 0,
    num_shards: int = 1,
) -> List[Path]:
    """Deterministically partition ``images`` via strided round-robin sharding.

    Strided slicing (``images[shard_idx::num_shards]``) balances archive
    prefixes and page densities across shards while guaranteeing exact
    disjointness and full union coverage across ``0 .. num_shards - 1``.
    """
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if shard_idx < 0 or shard_idx >= num_shards:
        raise ValueError(f"shard_idx must be in [0, {num_shards - 1}], got {shard_idx}")
    return list(images[shard_idx::num_shards])


def split_systems(raw_toks: Sequence[str]) -> List[List[str]]:
    """Split a full-page UniOMR token sequence on ``<sys>`` into non-empty systems."""
    systems: List[List[str]] = []
    cur: List[str] = []
    for tok in raw_toks:
        if tok == uni_omr.SYS_TOKEN:
            if cur:
                systems.append(cur)
                cur = []
        else:
            cur.append(tok)
    if cur:
        systems.append(cur)
    return systems


def safe_page_stem(image_path: str | Path) -> str:
    """Create a collision-free filesystem-safe stem for per-system bekern files."""
    p = Path(image_path)
    clean_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", p.stem).strip("._-") or "page"
    digest = hashlib.sha1(str(p).replace("\\", "/").encode("utf-8")).hexdigest()[:8]
    return f"{clean_stem[:80]}_{digest}"


def write_system_bekerns(
    systems: Sequence[Sequence[str]],
    bekern_dir: Path,
    page_stem: str,
) -> List[str]:
    """Write per-system ``*-bekern.txt`` files atomically and return their string paths."""
    bekern_dir.mkdir(parents=True, exist_ok=True)
    out_paths: List[str] = []
    for s_idx, sys_toks in enumerate(systems):
        out_p = bekern_dir / f"{page_stem}_s{s_idx}-bekern.txt"
        _atomic_write_text(out_p, " ".join(sys_toks) + "\n")
        out_paths.append(str(out_p))
    return out_paths


def evaluate_ov39_gate(
    *,
    mean_logprob: float,
    beat_valid_frac: float,
    r_parse: float,
    r_noloop: float,
    n_tokens: int,
    n_systems: int,
    cfg: Ov39GateConfig,
) -> Tuple[bool, str, float, List[str]]:
    """Evaluate the promoted ``ov-39`` gate and assign rhythm tier + sample weight.

    Returns ``(kept, tier, sample_weight, reject_reasons)``.
    """
    reasons: List[str] = []
    if float(mean_logprob) < float(cfg.min_logprob):
        reasons.append(f"low_logprob({mean_logprob:.4f}<{cfg.min_logprob:.2f})")
    if float(beat_valid_frac) < float(cfg.min_beat_valid):
        reasons.append(f"low_beat_valid({beat_valid_frac:.4f}<{cfg.min_beat_valid:.2f})")
    if float(r_parse) < float(cfg.min_r_parse):
        reasons.append(f"low_r_parse({r_parse:.4f}<{cfg.min_r_parse:.2f})")
    if float(r_noloop) < float(cfg.min_r_noloop):
        reasons.append(f"loop_detected(r_noloop={r_noloop:.4f}<{cfg.min_r_noloop:.2f})")
    if int(n_tokens) < int(cfg.min_tokens):
        reasons.append(f"too_short({n_tokens}<{cfg.min_tokens})")
    if int(n_systems) < 1:
        reasons.append("empty_systems")

    if reasons:
        return False, "rejected", 0.0, reasons

    if float(beat_valid_frac) >= float(cfg.high_rhythm_beat_valid):
        return True, "high_rhythm", float(cfg.high_rhythm_weight), []
    return True, "standard", float(cfg.standard_weight), []


def build_manifest_record(
    *,
    image_path: str | Path,
    bekern_paths: Sequence[str],
    tier: str,
    sample_weight: float,
    mean_logprob: float,
    beat_valid_frac: float,
    r_parse: float,
    r_noloop: float,
    n_tokens: int,
    n_systems: int,
    has_eos: bool,
    mtp_acc: float,
    max_dim: int,
    gen_max_len: int,
    ckpt_arch: str,
    ckpt_step: Optional[int],
    domain: str = "real_hw_pseudo",
) -> Dict[str, Any]:
    """Build a training-ready JSONL record matching ``uni_fullpage_iter11_clean.jsonl``."""
    return {
        "image": str(image_path),
        "bekerns": list(bekern_paths),
        "domain": str(domain),
        "tier": str(tier),
        "sample_weight": round(float(sample_weight), 4),
        "mean_logprob": round(float(mean_logprob), 5),
        "beat_valid_frac": round(float(beat_valid_frac), 5),
        "r_parse": round(float(r_parse), 4),
        "r_noloop": round(float(r_noloop), 4),
        "n_tokens": int(n_tokens),
        "n_systems": int(n_systems),
        "has_eos": bool(has_eos),
        "mtp_acc": round(float(mtp_acc), 4),
        "max_dim": int(max_dim),
        "gen_max_len": int(gen_max_len),
        "ckpt_arch": str(ckpt_arch),
        "ckpt_step": ckpt_step,
    }


def load_seen_images(*jsonl_paths: Path) -> Set[str]:
    """Load already-processed image paths from one or more append-only JSONL files."""
    seen: Set[str] = set()
    for p in jsonl_paths:
        if not p.is_file():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and "image" in obj:
                    seen.add(str(obj["image"]))
            except Exception:  # noqa: BLE001
                continue
    return seen


def merge_shard_manifests(
    shard_paths: Sequence[Path],
    out_path: Path,
    *,
    deduplicate: bool = True,
) -> Dict[str, Any]:
    """Merge multiple shard JSONL manifests into one deduplicated manifest."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen: Set[str] = set()
    kept_rows: List[Dict[str, Any]] = []
    n_dups = 0
    n_high = 0
    n_std = 0

    for sp in shard_paths:
        if not sp.is_file():
            raise FileNotFoundError(f"Shard manifest not found: {sp}")
        for line in sp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            img = str(rec["image"])
            if deduplicate and img in seen:
                n_dups += 1
                continue
            seen.add(img)
            kept_rows.append(rec)
            if rec.get("tier") == "high_rhythm":
                n_high += 1
            else:
                n_std += 1

    tmp = out_path.with_name(out_path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for rec in kept_rows:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, out_path)

    return {
        "out_manifest": str(out_path),
        "n_shards": len(shard_paths),
        "n_kept": len(kept_rows),
        "n_high_rhythm": n_high,
        "n_standard": n_std,
        "n_duplicates_skipped": n_dups,
    }


# ---------------------------------------------------------------------------
# Checkpoint-Agnostic Loader (UniOMR v1 + UniOMR v2) & Precision Guard
# ---------------------------------------------------------------------------

def detect_checkpoint_arch(ck: Any) -> str:
    """Detect whether a loaded checkpoint dict belongs to ``v1`` or ``v2``."""
    if not isinstance(ck, dict):
        return "v1"
    arch_hint = str(ck.get("arch", ck.get("model_arch", ""))).strip().lower()
    if arch_hint in ("v2", "uni_omr_v2", "uniomrv2", "uniomrv2net"):
        return "v2"
    if arch_hint in ("v1", "uni_omr", "uniomr", "uniomrnet"):
        return "v1"

    cfg_dict = ck.get("config") or ck.get("cfg") or {}
    if isinstance(cfg_dict, dict) and any(k in _V2_CFG_MARKERS for k in cfg_dict.keys()):
        return "v2"

    sd = ck.get("model") if "model" in ck else ck
    if isinstance(sd, dict):
        for k in sd.keys():
            if any(str(k).startswith(pfx) for pfx in _V2_SD_PREFIXES):
                return "v2"
    return "v1"


def load_any_uni_checkpoint(
    ckpt_path: str | Path,
    device: torch.device,
) -> Tuple[torch.nn.Module, Any, Dict[str, Any]]:
    """Load a ``UniOMRNet`` (v1) or ``UniOMRv2Net`` (v2) checkpoint onto ``device``."""
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    arch = detect_checkpoint_arch(ck)
    cfg_dict = (ck.get("config") or ck.get("cfg") or {}) if isinstance(ck, dict) else {}
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck

    if arch == "v2":
        valid_fields = {f.name for f in fields(uni_omr_v2.UniOMRv2Config)}
        filtered_cfg = {k: v for k, v in cfg_dict.items() if k in valid_fields}
        cfg = uni_omr_v2.UniOMRv2Config(**filtered_cfg)
        model: torch.nn.Module = uni_omr_v2.UniOMRv2Net(cfg)
    else:
        valid_fields = {f.name for f in fields(uni_omr.UniOMRConfig)}
        filtered_cfg = {k: v for k, v in cfg_dict.items() if k in valid_fields}
        cfg = uni_omr.UniOMRConfig(**filtered_cfg)
        model = uni_omr.UniOMRNet(cfg)

    inc = model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    meta = {
        "arch": arch,
        "step": ck.get("step") if isinstance(ck, dict) else None,
        "val_ser": ck.get("val_ser", ck.get("best_val_ser")) if isinstance(ck, dict) else None,
        "missing_keys": list(inc.missing_keys),
        "unexpected_keys": list(inc.unexpected_keys),
    }
    return model, cfg, meta


def resolve_bf16_support(requested_bf16: bool, device: torch.device) -> Tuple[bool, str]:
    """Return ``(use_bf16, reason)`` with automatic T4 / CPU FP32 fallback."""
    if not requested_bf16:
        return False, "fp32 (bf16 not requested)"
    if device.type != "cuda":
        return False, f"fp32 (device={device.type})"
    if not torch.cuda.is_available():
        return False, "fp32 (cuda unavailable)"
    cc = torch.cuda.get_device_capability(device)
    if cc[0] >= 8:
        return True, f"bf16 (cc={cc[0]}.{cc[1]} >= 8.0)"
    return False, f"fp32 fallback (cc={cc[0]}.{cc[1]} < 8.0; T4-safe)"


def configure_cuda_sdpa(device: torch.device) -> None:
    """Configure CUTLASS mem-efficient / flash SDPA (never cuDNN SDPA)."""
    if device.type == "cuda" and torch.cuda.is_available():
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


# ---------------------------------------------------------------------------
# Single-Page Decode & Exact ov-39 Scoring
# ---------------------------------------------------------------------------

def decode_and_score_page(
    model: torch.nn.Module,
    cfg: Any,
    img_path: Path,
    device: torch.device,
    *,
    max_dim: int = 1280,
    gen_max_len: int = 2048,
    use_bf16: bool = False,
    strict_grammar: bool = False,
) -> Dict[str, Any]:
    """Decode one RISM page with FSA + MTP-4 and compute exact ``ov-39`` scores."""
    _, _, eos_id, _, _, id_to_tok = ev._vocab_spec(int(cfg.vocab_size))

    with Image.open(img_path) as raw_im:
        pil_im = uni_omr._resize_fullpage_pil(raw_im.convert("RGB"), int(max_dim))
    img_t = uni_omr._pil_to_chw_tensor(pil_im).unsqueeze(0).to(device)

    t0 = time.perf_counter()
    ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if (use_bf16 and device.type == "cuda")
        else nullcontext()
    )
    with torch.no_grad():
        with ctx:
            seq, mtp_stats = ev.generate_mtp_spec(
                model,
                img_t,
                max_len=int(gen_max_len),
                use_grammar=True,
                strict_grammar=bool(strict_grammar),
                return_stats=True,
            )
            if seq.shape[1] >= 2:
                out = model(img_t, seq[:, :-1], return_aux=False, return_mtp=False)
                lp = F.log_softmax(out["logits"].float(), dim=-1)
                tgt = seq[:, 1:]
                tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                mean_lp = float(tok_lp.mean().item())
            else:
                mean_lp = -10.0

    elapsed = max(1e-6, time.perf_counter() - t0)
    seq_ids = seq[0].detach().cpu().tolist()
    has_eos = eos_id in seq_ids[1:]
    pred_raw = ev.decode_uni_compat(
        seq_ids, vocab_size=int(cfg.vocab_size), skip_special=True
    )
    if not has_eos:
        pred_raw = ev.trim_truncated_tokens(pred_raw)

    struct = rlvr_grpo.compute_unlabeled_structural_reward(
        pred_raw, vocab=id_to_tok, terminated_eos=has_eos
    )
    systems = split_systems(pred_raw)

    return {
        "raw_tokens": pred_raw,
        "systems": systems,
        "n_tokens": len(pred_raw),
        "n_systems": len(systems),
        "mean_logprob": round(mean_lp, 5),
        "beat_valid_frac": round(float(struct["r_rhythm"]), 5),
        "r_parse": round(float(struct["r_parse"]), 4),
        "r_noloop": round(float(struct["r_noloop"]), 4),
        "has_eos": bool(has_eos),
        "mtp_acc": round(float(mtp_stats.get("acceptance_rate", 0.0)), 4),
        "elapsed_secs": round(elapsed, 4),
    }


# ---------------------------------------------------------------------------
# Resumable Sharded Pipeline Runner
# ---------------------------------------------------------------------------

def run_pipeline(
    args: argparse.Namespace,
    *,
    decode_fn: Optional[Callable[[Path], Dict[str, Any]]] = None,
    ckpt_meta_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the resumable, sharded ``ov-39`` RISM pseudo-label pipeline.

    ``decode_fn`` and ``ckpt_meta_override`` allow fast deterministic CPU unit
    tests without instantiating a full neural checkpoint.
    """
    input_dir = Path(args.input_dir)
    out_manifest = Path(args.out)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    bekern_dir = Path(args.bekern_dir) if args.bekern_dir else (out_manifest.parent / "bekerns")
    scored_ledger = out_manifest.with_name(out_manifest.stem + ".all_scored.jsonl")
    summary_path = out_manifest.with_name(out_manifest.stem + ".summary.json")

    all_discovered = discover_images(input_dir)
    shard_imgs = shard_images(
        all_discovered,
        shard_idx=int(args.shard_idx),
        num_shards=int(args.num_shards),
    )
    if args.limit is not None:
        shard_imgs = shard_imgs[: int(args.limit)]

    gate_cfg = Ov39GateConfig(
        min_logprob=float(args.min_logprob),
        min_beat_valid=float(args.min_beat_valid),
        high_rhythm_beat_valid=float(args.high_rhythm_beat_valid),
        min_r_parse=float(args.min_r_parse),
        min_r_noloop=float(args.min_r_noloop),
        min_tokens=int(args.min_tokens),
        high_rhythm_weight=float(args.high_rhythm_weight),
        domain=str(args.domain),
    )

    if decode_fn is None:
        device = torch.device(args.device)
        configure_cuda_sdpa(device)
        use_bf16, prec_reason = resolve_bf16_support(bool(args.bf16), device)
        model, model_cfg, ckpt_meta = load_any_uni_checkpoint(args.ckpt, device=device)

        def _default_decode(p: Path) -> Dict[str, Any]:
            return decode_and_score_page(
                model,
                model_cfg,
                p,
                device,
                max_dim=int(args.max_dim),
                gen_max_len=int(args.gen_max_len),
                use_bf16=use_bf16,
                strict_grammar=bool(args.strict_grammar),
            )

        decode_fn = _default_decode
    else:
        prec_reason = "custom-decode-fn"
        ckpt_meta = dict(ckpt_meta_override or {"arch": "mock", "step": 0})

    if args.resume:
        seen = load_seen_images(out_manifest, scored_ledger)
    else:
        seen = set()
        out_manifest.write_text("", encoding="utf-8")
        scored_ledger.write_text("", encoding="utf-8")

    print(
        f"[RISM ov-39] ckpt={args.ckpt} (arch={ckpt_meta.get('arch')}, step={ckpt_meta.get('step')}) | "
        f"prec={prec_reason} | shard={args.shard_idx}/{args.num_shards} | "
        f"pages={len(shard_imgs)}/{len(all_discovered)} (resume_skip={len(seen)}) | "
        f"max_dim={args.max_dim} gen_max_len={args.gen_max_len} | "
        f"gate=(lp>={gate_cfg.min_logprob:g}, beat>={gate_cfg.min_beat_valid:g}, "
        f"high_rhythm>={gate_cfg.high_rhythm_beat_valid:g})",
        flush=True,
    )

    counts = {
        "total_discovered": len(all_discovered),
        "shard_total": len(shard_imgs),
        "skipped_resume": 0,
        "processed_this_run": 0,
        "kept_this_run": 0,
        "high_rhythm_this_run": 0,
        "standard_this_run": 0,
        "rejected_this_run": 0,
        "errors_this_run": 0,
    }
    reject_hist: Dict[str, int] = {}
    t_start = time.perf_counter()

    with out_manifest.open("a", encoding="utf-8") as mf, scored_ledger.open("a", encoding="utf-8") as lf:
        for idx, img_path in enumerate(shard_imgs, start=1):
            img_str = str(img_path)
            if img_str in seen:
                counts["skipped_resume"] += 1
                continue

            try:
                res = decode_fn(img_path)
                kept, tier, weight, reasons = evaluate_ov39_gate(
                    mean_logprob=float(res["mean_logprob"]),
                    beat_valid_frac=float(res["beat_valid_frac"]),
                    r_parse=float(res["r_parse"]),
                    r_noloop=float(res["r_noloop"]),
                    n_tokens=int(res["n_tokens"]),
                    n_systems=int(res["n_systems"]),
                    cfg=gate_cfg,
                )

                counts["processed_this_run"] += 1
                ledger_row = {
                    "image": img_str,
                    "kept": bool(kept),
                    "tier": tier,
                    "sample_weight": round(float(weight), 4),
                    "mean_logprob": round(float(res["mean_logprob"]), 5),
                    "beat_valid_frac": round(float(res["beat_valid_frac"]), 5),
                    "r_parse": round(float(res["r_parse"]), 4),
                    "r_noloop": round(float(res["r_noloop"]), 4),
                    "n_tokens": int(res["n_tokens"]),
                    "n_systems": int(res["n_systems"]),
                    "has_eos": bool(res["has_eos"]),
                    "mtp_acc": round(float(res.get("mtp_acc", 0.0)), 4),
                    "reject_reasons": reasons,
                }

                if kept:
                    p_stem = safe_page_stem(img_path)
                    bek_paths = write_system_bekerns(res["systems"], bekern_dir, p_stem)
                    rec = build_manifest_record(
                        image_path=img_str,
                        bekern_paths=bek_paths,
                        tier=tier,
                        sample_weight=weight,
                        mean_logprob=float(res["mean_logprob"]),
                        beat_valid_frac=float(res["beat_valid_frac"]),
                        r_parse=float(res["r_parse"]),
                        r_noloop=float(res["r_noloop"]),
                        n_tokens=int(res["n_tokens"]),
                        n_systems=int(res["n_systems"]),
                        has_eos=bool(res["has_eos"]),
                        mtp_acc=float(res.get("mtp_acc", 0.0)),
                        max_dim=int(args.max_dim),
                        gen_max_len=int(args.gen_max_len),
                        ckpt_arch=str(ckpt_meta.get("arch", "unknown")),
                        ckpt_step=ckpt_meta.get("step"),
                        domain=gate_cfg.domain,
                    )
                    mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    mf.flush()
                    counts["kept_this_run"] += 1
                    if tier == "high_rhythm":
                        counts["high_rhythm_this_run"] += 1
                    else:
                        counts["standard_this_run"] += 1
                else:
                    counts["rejected_this_run"] += 1
                    for r in reasons:
                        key = r.split("(", 1)[0]
                        reject_hist[key] = reject_hist.get(key, 0) + 1

                lf.write(json.dumps(ledger_row, ensure_ascii=False) + "\n")
                lf.flush()
                seen.add(img_str)

                if idx % max(1, int(args.log_every)) == 0 or idx == len(shard_imgs):
                    elapsed = time.perf_counter() - t_start
                    rate = counts["processed_this_run"] / max(1e-6, elapsed)
                    yield_pct = 100.0 * counts["kept_this_run"] / max(1, counts["processed_this_run"])
                    print(
                        f"[shard {args.shard_idx}/{args.num_shards} {idx:04d}/{len(shard_imgs):04d}] "
                        f"kept={counts['kept_this_run']} (hi={counts['high_rhythm_this_run']}, "
                        f"yield={yield_pct:.1f}%) | {rate:.2f} pg/s | "
                        f"last={tier} lp={res['mean_logprob']:.3f} beat={res['beat_valid_frac']:.2f} "
                        f"n={res['n_tokens']} eos={int(res['has_eos'])}",
                        flush=True,
                    )
            except Exception as exc:  # noqa: BLE001
                counts["errors_this_run"] += 1
                print(
                    f"[shard {args.shard_idx}/{args.num_shards} {idx:04d}/{len(shard_imgs):04d}] "
                    f"{img_path.name} ERROR: {exc}",
                    flush=True,
                )

    # Aggregate cumulative totals across resumed runs from the JSONL files.
    cum_kept = 0
    cum_high = 0
    cum_std = 0
    cum_eos = 0
    if out_manifest.is_file():
        for line in out_manifest.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            cum_kept += 1
            if r.get("tier") == "high_rhythm":
                cum_high += 1
            else:
                cum_std += 1
            if r.get("has_eos"):
                cum_eos += 1
    cum_scored = 0
    if scored_ledger.is_file():
        cum_scored = sum(1 for ln in scored_ledger.read_text(encoding="utf-8").splitlines() if ln.strip())

    elapsed_total = round(time.perf_counter() - t_start, 2)
    summary = {
        "completed_utc": _utc_now(),
        "ckpt": str(args.ckpt),
        "ckpt_meta": ckpt_meta,
        "precision": prec_reason,
        "input_dir": str(input_dir),
        "out_manifest": str(out_manifest),
        "bekern_dir": str(bekern_dir),
        "scored_ledger": str(scored_ledger),
        "shard_idx": int(args.shard_idx),
        "num_shards": int(args.num_shards),
        "max_dim": int(args.max_dim),
        "gen_max_len": int(args.gen_max_len),
        "gate_config": asdict(gate_cfg),
        "counts": counts,
        "cumulative": {
            "total_scored": cum_scored,
            "total_kept": cum_kept,
            "total_high_rhythm": cum_high,
            "total_standard": cum_std,
            "yield_pct": round(100.0 * cum_kept / max(1, cum_scored), 2),
            "eos_rate_kept": round(cum_eos / max(1, cum_kept), 4),
        },
        "reject_reason_hist": reject_hist,
        "elapsed_secs": elapsed_total,
    }
    _atomic_write_json(summary_path, summary)
    print(
        f"[RISM ov-39] Done in {elapsed_total}s | kept={cum_kept}/{cum_scored} "
        f"({summary['cumulative']['yield_pct']:.1f}%, high_rhythm={cum_high}) -> {out_manifest}",
        flush=True,
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Promoted ov-39 RISM Pseudo-Label Manifest Builder (UniOMR v1 & v2)."
    )
    ap.add_argument(
        "--ckpt",
        default="",
        help="Path to UniOMR v1 or v2 teacher checkpoint (.pt). Required unless --merge-shards is used.",
    )
    ap.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help=f"Directory of unlabeled RISM pages (default: {DEFAULT_INPUT_DIR}).",
    )
    ap.add_argument(
        "--out",
        default=str(DEFAULT_OUT_MANIFEST),
        help=f"Output training-ready JSONL manifest path (default: {DEFAULT_OUT_MANIFEST}).",
    )
    ap.add_argument(
        "--bekern-dir",
        default=None,
        help="Directory for per-system *-bekern.txt files (default: <out_dir>/bekerns).",
    )
    ap.add_argument("--device", default="cuda", help="Inference device ('cuda' or 'cpu').")
    ap.add_argument(
        "--bf16",
        action="store_true",
        help="Enable bfloat16 autocast on Ampere+ CUDA GPUs (auto-falls back to FP32 on T4/CPU).",
    )
    ap.add_argument("--max-dim", type=int, default=1280, help="Full-page resize max_dim (default: 1280).")
    ap.add_argument("--gen-max-len", type=int, default=2048, help="Max generated tokens per page (default: 2048).")
    ap.add_argument("--strict-grammar", action="store_true", help="Use strict=True FSA PrefixState.")

    # Promoted ov-39 thresholds
    ap.add_argument("--min-logprob", type=float, default=-0.35, help="Min mean_logprob gate (ov-39: -0.35).")
    ap.add_argument("--min-beat-valid", type=float, default=0.70, help="Min beat_valid_frac gate (ov-39: 0.70).")
    ap.add_argument(
        "--high-rhythm-beat-valid",
        type=float,
        default=0.85,
        help="Threshold for tier='high_rhythm' tag (ov-39: 0.85).",
    )
    ap.add_argument(
        "--high-rhythm-weight",
        type=float,
        default=1.75,
        help="Sampling weight for tier='high_rhythm' rows (ov-39 recommended 1.5-2.0; default: 1.75).",
    )
    ap.add_argument("--min-r-parse", type=float, default=0.50, help="Min r_parse structural score (ov-39: 0.50).")
    ap.add_argument("--min-r-noloop", type=float, default=0.99, help="Min r_noloop score (ov-39: 1.0 / >=0.99).")
    ap.add_argument("--min-tokens", type=int, default=24, help="Min decoded tokens per page (ov-39: 24).")
    ap.add_argument("--domain", default="real_hw_pseudo", help="Manifest domain tag (default: real_hw_pseudo).")

    # Sharding & resume
    ap.add_argument("--shard-idx", type=int, default=0, help="0-based shard index for parallel workers.")
    ap.add_argument("--num-shards", type=int, default=1, help="Total number of parallel shards.")
    ap.add_argument("--resume", action="store_true", help="Append-only resume skipping already-scored pages.")
    ap.add_argument("--limit", type=int, default=None, help="Optional cap on pages processed in this shard.")
    ap.add_argument("--log-every", type=int, default=25, help="Progress log cadence.")

    # Shard merge utility
    ap.add_argument(
        "--merge-shards",
        nargs="+",
        default=None,
        help="Merge completed shard JSONL manifests into --out and exit.",
    )
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.merge_shards:
        res = merge_shard_manifests([Path(p) for p in args.merge_shards], Path(args.out))
        print(json.dumps(res, indent=2), flush=True)
        return
    if not args.ckpt:
        raise SystemExit("Error: --ckpt is required unless --merge-shards is provided.")
    run_pipeline(args)


if __name__ == "__main__":
    main()
