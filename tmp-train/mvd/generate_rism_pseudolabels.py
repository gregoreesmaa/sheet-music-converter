#!/usr/bin/env python3
"""Multi-View + FSA-Gated Noisy Student RISM Pseudo-Label Generator.

Runs a trained ``UniOMRNet`` teacher checkpoint (MTP-4 self-speculative decoding
+ ``PrefixState`` FSA grammar) on **2 views** of each unlabeled RISM archival
handwritten page:

  - View 1: native ``--max-dim`` (default 1280)
  - View 2: perturbed ``0.93x`` scale + mild contrast shift (``0.90x``)

Applies strict Quality >> Quantity gates:
  1. Both views pass ``bekern-decode`` (``DecOK == True``, zero malformed
     measures) AND ``n_sys`` matches between View 1 and View 2.
  2. Ordered ``TokLCS F1(view1, view2) >= --min-view-agree`` (default 0.85).
  3. Mean MTP acceptance rate ``>= --min-mtp-acc`` (default 0.75) and
     ``--min-tokens <= len <= --max-tokens`` for both views.
  4. Optional system-level salvage mode (``--salvage-systems``): even if a full
     page fails global agreement, extract individual matching ``<sys>`` strips
     where View 1 and View 2 both pass ``bekern-decode`` and agree
     ``TokLCS F1 >= --min-sys-agree`` (default 0.90).

Outputs a resumable JSONL manifest (default ``D:\\tmp\\rism-pseudolabels\\manifest.jsonl``)
with full confidence metadata plus an append-only processed-image ledger so
interrupted runs resume cleanly without re-evaluating rejected pages.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from PIL import Image, ImageEnhance
import torch

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import eval_sys_gate as esg  # noqa: E402
import eval_uni_suite31 as ev  # noqa: E402
import uni_omr  # noqa: E402


DEFAULT_INPUT_DIR = Path(r"D:\tmp\unlabeled-10k-real\rism_hw")
DEFAULT_OUT_MANIFEST = Path(r"D:\tmp\rism-pseudolabels\manifest.jsonl")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp"}


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class ViewPrediction:
    """Decoded prediction and confidence stats for one view of a page."""

    raw_tokens: List[str]
    stitched_tokens: List[str]
    systems: List[List[str]]
    n_sys: int
    decode_ok: bool
    mtp_acc: float
    truncated: bool = False
    elapsed_secs: float = 0.0


@dataclass
class GateConfig:
    """Quality gates for 2-view consistency + FSA/MTP pseudo-label filtering."""

    min_view_agree: float = 0.85
    min_mtp_acc: float = 0.75
    min_tokens: int = 24
    max_tokens: int = 1950
    salvage_systems: bool = False
    min_sys_agree: float = 0.90
    min_sys_tokens: int = 8


# ---------------------------------------------------------------------------
# Core Pure Helpers (CPU-Testable)
# ---------------------------------------------------------------------------

def split_systems(raw_toks: Sequence[str]) -> List[List[str]]:
    """Split a full-page UniOMR token list on ``<sys>`` into non-empty systems."""
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


def make_view_images(
    pil_im: Image.Image,
    max_dim: int = 1280,
    view2_scale: float = 0.93,
    view2_contrast: float = 0.90,
) -> Tuple[Image.Image, Image.Image]:
    """Build View 1 (native ``max_dim``) and View 2 (scaled + contrast-shifted)."""
    rgb = pil_im.convert("RGB")
    v1 = uni_omr._resize_fullpage_pil(rgb, int(max_dim))
    v2_dim = max(64, int(round(float(max_dim) * float(view2_scale))))
    v2_resized = uni_omr._resize_fullpage_pil(rgb, v2_dim)
    v2 = ImageEnhance.Contrast(v2_resized).enhance(float(view2_contrast))
    return v1, v2


def check_bekern_decode(
    toks: Sequence[str],
    workdir: Path,
    stem: str,
) -> bool:
    """Return True iff ``assembly-serialize bekern-decode`` succeeds with valid notes."""
    if not toks:
        return False
    hyp_txt = workdir / f"{stem}-bekern.txt"
    hyp_notes = workdir / f"{stem}-notes.json"
    hyp_txt.write_text(" ".join(toks) + "\n", encoding="utf-8")
    if hyp_notes.exists():
        hyp_notes.unlink()
    proc = esg.run_cli([
        esg.ASER,
        "bekern-decode",
        "--input",
        str(hyp_txt),
        "--out",
        str(hyp_notes),
    ])
    if proc.returncode != 0 or not hyp_notes.is_file():
        return False
    try:
        payload = json.loads(hyp_notes.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return False
    if isinstance(payload, dict):
        parts = payload.get("parts")
        if isinstance(parts, list):
            n_notes = sum(
                len(p.get("notes", []))
                for p in parts
                if isinstance(p, dict) and isinstance(p.get("notes"), list)
            )
            if n_notes > 0:
                return True
        notes = payload.get("notes")
        if isinstance(notes, list) and len(notes) > 0:
            return True
        measures = payload.get("measures")
        if isinstance(measures, list) and len(measures) > 0:
            return True
        return False
    if isinstance(payload, list):
        return len(payload) > 0
    return False


def salvage_matching_systems(
    v1_systems: Sequence[Sequence[str]],
    v2_systems: Sequence[Sequence[str]],
    cfg: GateConfig,
    sys_decode_fn: Optional[Callable[[Sequence[str]], bool]] = None,
) -> List[Dict[str, Any]]:
    """Extract individual ``<sys>`` strips where View 1 & View 2 agree >= ``min_sys_agree``.

    Pairs systems 1-to-1 by index when ``len(v1_systems) == len(v2_systems)``,
    or by greedy highest-TokLCS matching when system counts differ. Each
    salvaged strip must satisfy:
      - ``len(s1) >= cfg.min_sys_tokens`` and ``len(s2) >= cfg.min_sys_tokens``
      - ``compute_token_lcs_f1(s1, s2) >= cfg.min_sys_agree``
      - ``sys_decode_fn(s1)`` and ``sys_decode_fn(s2)`` both True (when provided)
    """
    if not v1_systems or not v2_systems:
        return []

    candidate_pairs: List[Tuple[int, int, float]] = []
    if len(v1_systems) == len(v2_systems):
        for i, (s1, s2) in enumerate(zip(v1_systems, v2_systems)):
            lcs = ev.compute_token_lcs_f1(s1, s2)
            candidate_pairs.append((i, i, lcs))
    else:
        scored: List[Tuple[float, int, int]] = []
        for i, s1 in enumerate(v1_systems):
            for j, s2 in enumerate(v2_systems):
                lcs = ev.compute_token_lcs_f1(s1, s2)
                if lcs >= cfg.min_sys_agree:
                    scored.append((lcs, i, j))
        scored.sort(key=lambda x: x[0], reverse=True)
        used_v1: Set[int] = set()
        used_v2: Set[int] = set()
        for lcs, i, j in scored:
            if i in used_v1 or j in used_v2:
                continue
            used_v1.add(i)
            used_v2.add(j)
            candidate_pairs.append((i, j, lcs))
        candidate_pairs.sort(key=lambda x: x[0])

    salvaged: List[Dict[str, Any]] = []
    for i, j, lcs in candidate_pairs:
        s1 = list(v1_systems[i])
        s2 = list(v2_systems[j])
        if len(s1) < cfg.min_sys_tokens or len(s2) < cfg.min_sys_tokens:
            continue
        if lcs < cfg.min_sys_agree:
            continue
        if sys_decode_fn is not None:
            if not (sys_decode_fn(s1) and sys_decode_fn(s2)):
                continue
        salvaged.append({
            "sys_idx_v1": int(i),
            "sys_idx_v2": int(j),
            "tok_lcs": round(float(lcs), 6),
            "n_toks": len(s1),
            "tokens": s1,
        })
    return salvaged


def evaluate_two_views(
    image_path: str | Path,
    v1: ViewPrediction,
    v2: ViewPrediction,
    cfg: GateConfig,
    sys_decode_fn: Optional[Callable[[Sequence[str]], bool]] = None,
) -> Dict[str, Any]:
    """Apply all strict quality gates to a 2-view prediction pair.

    Returns a record dict with ``status`` in ``{"accepted", "salvaged", "rejected"}``
    and full confidence metadata.
    """
    view_tok_lcs = ev.compute_token_lcs_f1(v1.stitched_tokens, v2.stitched_tokens)
    mean_mtp_acc = 0.5 * (float(v1.mtp_acc) + float(v2.mtp_acc))
    v1_len = len(v1.stitched_tokens)
    v2_len = len(v2.stitched_tokens)

    reject_reasons: List[str] = []
    if not v1.decode_ok:
        reject_reasons.append("v1_decode_fail")
    if not v2.decode_ok:
        reject_reasons.append("v2_decode_fail")
    if v1.n_sys < 1 or v2.n_sys < 1:
        reject_reasons.append("empty_systems")
    elif v1.n_sys != v2.n_sys:
        reject_reasons.append(f"n_sys_mismatch({v1.n_sys}!={v2.n_sys})")
    if view_tok_lcs < cfg.min_view_agree:
        reject_reasons.append(f"low_view_agree({view_tok_lcs:.3f}<{cfg.min_view_agree:.3f})")
    if mean_mtp_acc < cfg.min_mtp_acc:
        reject_reasons.append(f"low_mtp_acc({mean_mtp_acc:.3f}<{cfg.min_mtp_acc:.3f})")
    if v1_len < cfg.min_tokens or v2_len < cfg.min_tokens:
        reject_reasons.append(f"too_short({v1_len},{v2_len}<{cfg.min_tokens})")
    if v1_len > cfg.max_tokens or v2_len > cfg.max_tokens:
        reject_reasons.append(f"too_long({v1_len},{v2_len}>{cfg.max_tokens})")

    salvaged_systems: List[Dict[str, Any]] = []
    if not reject_reasons:
        status = "accepted"
    elif cfg.salvage_systems and mean_mtp_acc >= cfg.min_mtp_acc:
        salvaged_systems = salvage_matching_systems(
            v1.systems,
            v2.systems,
            cfg=cfg,
            sys_decode_fn=sys_decode_fn,
        )
        status = "salvaged" if salvaged_systems else "rejected"
    else:
        status = "rejected"

    record: Dict[str, Any] = {
        "image": str(image_path),
        "domain": "rism_hw_pseudo",
        "status": status,
        "view_tok_lcs": round(float(view_tok_lcs), 6),
        "mean_mtp_acc": round(float(mean_mtp_acc), 6),
        "v1_mtp_acc": round(float(v1.mtp_acc), 6),
        "v2_mtp_acc": round(float(v2.mtp_acc), 6),
        "v1_decode_ok": bool(v1.decode_ok),
        "v2_decode_ok": bool(v2.decode_ok),
        "v1_n_sys": int(v1.n_sys),
        "v2_n_sys": int(v2.n_sys),
        "v1_n_toks": int(v1_len),
        "v2_n_toks": int(v2_len),
        "v1_truncated": bool(v1.truncated),
        "v2_truncated": bool(v2.truncated),
        "reject_reasons": reject_reasons,
    }
    if status == "accepted":
        record["tokens"] = list(v1.stitched_tokens)
        record["raw_tokens"] = list(v1.raw_tokens)
        record["systems"] = [list(s) for s in v1.systems]
    elif status == "salvaged":
        record["salvaged_systems"] = salvaged_systems
    return record


# ---------------------------------------------------------------------------
# Teacher Decoding & Resumable Pipeline
# ---------------------------------------------------------------------------

def discover_images(input_dir: Path) -> List[Path]:
    """Find all image files under ``input_dir`` in deterministic sorted order."""
    if input_dir.is_file() and input_dir.suffix.lower() in IMAGE_EXTS:
        return [input_dir]
    imgs = [
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    imgs.sort(key=lambda p: str(p).lower())
    return imgs


def _decode_single_view(
    model: uni_omr.UniOMRNet,
    cfg: uni_omr.UniOMRConfig,
    pil_im: Image.Image,
    device: torch.device,
    max_len: int,
    use_grammar: bool,
    strict_grammar: bool,
    measure_beam_size: int,
    use_bf16: bool,
    workdir: Path,
    stem: str,
) -> ViewPrediction:
    img_t = uni_omr._pil_to_chw_tensor(pil_im).unsqueeze(0).to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if (use_bf16 and device.type == "cuda")
            else ev._nullcontext()
        )
        with ctx:
            seq, mtp_stats = ev.generate_mtp_spec(
                model,
                img_t,
                max_len=max_len,
                use_grammar=use_grammar,
                strict_grammar=strict_grammar,
                return_stats=True,
                measure_beam_size=measure_beam_size,
            )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = max(1e-6, time.perf_counter() - t0)

    seq_ids = seq[0].detach().cpu().tolist()
    _, _, eos_id, _, _, _ = ev._vocab_spec(int(cfg.vocab_size))
    truncated = eos_id not in seq_ids[1:]
    raw_toks = ev.decode_uni_compat(
        seq_ids, vocab_size=int(cfg.vocab_size), skip_special=True
    )
    if truncated:
        raw_toks = ev.trim_truncated_tokens(raw_toks)
    stitched_toks, n_sys = ev.stitch_uni_tokens(raw_toks)
    systems = split_systems(raw_toks)
    dec_ok = check_bekern_decode(stitched_toks, workdir=workdir, stem=stem)
    return ViewPrediction(
        raw_tokens=raw_toks,
        stitched_tokens=stitched_toks,
        systems=systems,
        n_sys=n_sys,
        decode_ok=dec_ok,
        mtp_acc=float(mtp_stats.get("acceptance_rate", 0.0)),
        truncated=truncated,
        elapsed_secs=round(elapsed, 4),
    )


def _load_seen_images(out_manifest: Path, processed_log: Path) -> Set[str]:
    seen: Set[str] = set()
    if out_manifest.is_file():
        for line in out_manifest.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "image" in obj:
                    seen.add(str(obj["image"]))
            except Exception:  # noqa: BLE001
                continue
    if processed_log.is_file():
        for line in processed_log.read_text(encoding="utf-8").splitlines():
            p = line.strip()
            if p:
                seen.add(p)
    return seen


def run_generation(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device(args.device)
    model, model_cfg, ckpt_meta = ev.load_uni_checkpoint(args.ckpt, device=device)

    input_dir = Path(args.input_dir)
    out_manifest = Path(args.out)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    processed_log = out_manifest.with_suffix(out_manifest.suffix + ".processed.txt")

    all_images = discover_images(input_dir)
    if args.limit is not None:
        all_images = all_images[: int(args.limit)]

    seen = _load_seen_images(out_manifest, processed_log) if args.resume else set()
    if not args.resume:
        out_manifest.write_text("", encoding="utf-8")
        processed_log.write_text("", encoding="utf-8")

    gate_cfg = GateConfig(
        min_view_agree=float(args.min_view_agree),
        min_mtp_acc=float(args.min_mtp_acc),
        min_tokens=int(args.min_tokens),
        max_tokens=int(args.max_tokens),
        salvage_systems=bool(args.salvage_systems),
        min_sys_agree=float(args.min_sys_agree),
        min_sys_tokens=int(args.min_sys_tokens),
    )

    print(
        f"[RISM Pseudo-Label] ckpt={args.ckpt} (step={ckpt_meta['step']}) | "
        f"device={device} | images={len(all_images)} (resume_skip={len(seen)}) | "
        f"max_dim={args.max_dim} | view2=({args.view2_scale:g}x, contrast={args.view2_contrast:g}) | "
        f"min_agree={gate_cfg.min_view_agree:g} | min_mtp={gate_cfg.min_mtp_acc:g} | "
        f"salvage={gate_cfg.salvage_systems}",
        flush=True,
    )

    counts = {
        "total_discovered": len(all_images),
        "skipped_resume": 0,
        "processed": 0,
        "accepted": 0,
        "salvaged_pages": 0,
        "salvaged_systems": 0,
        "rejected": 0,
        "errors": 0,
    }
    reason_hist: Dict[str, int] = {}

    workdir = Path(tempfile.mkdtemp(prefix="rism_pseudo_"))
    t_start = time.perf_counter()

    with out_manifest.open("a", encoding="utf-8") as mf, processed_log.open("a", encoding="utf-8") as pf:
        for idx, img_path in enumerate(all_images, start=1):
            img_str = str(img_path)
            if img_str in seen:
                counts["skipped_resume"] += 1
                continue

            try:
                with Image.open(img_path) as raw_im:
                    v1_im, v2_im = make_view_images(
                        raw_im,
                        max_dim=int(args.max_dim),
                        view2_scale=float(args.view2_scale),
                        view2_contrast=float(args.view2_contrast),
                    )
                stem = f"p{idx:06d}"
                v1 = _decode_single_view(
                    model=model,
                    cfg=model_cfg,
                    pil_im=v1_im,
                    device=device,
                    max_len=int(args.max_len),
                    use_grammar=True,
                    strict_grammar=bool(args.strict_grammar),
                    measure_beam_size=int(args.measure_beam_size),
                    use_bf16=bool(args.bf16),
                    workdir=workdir,
                    stem=f"{stem}-v1",
                )
                v2 = _decode_single_view(
                    model=model,
                    cfg=model_cfg,
                    pil_im=v2_im,
                    device=device,
                    max_len=int(args.max_len),
                    use_grammar=True,
                    strict_grammar=bool(args.strict_grammar),
                    measure_beam_size=int(args.measure_beam_size),
                    use_bf16=bool(args.bf16),
                    workdir=workdir,
                    stem=f"{stem}-v2",
                )

                sys_dec_ctr = {"n": 0}

                def _sys_decode(toks: Sequence[str]) -> bool:
                    sys_dec_ctr["n"] += 1
                    return check_bekern_decode(
                        esg.canonicalize_bekern_tail_order(list(toks)),
                        workdir=workdir,
                        stem=f"{stem}-s{sys_dec_ctr['n']}",
                    )

                rec = evaluate_two_views(
                    image_path=img_path,
                    v1=v1,
                    v2=v2,
                    cfg=gate_cfg,
                    sys_decode_fn=_sys_decode,
                )
                rec["ckpt"] = str(args.ckpt)
                rec["ckpt_step"] = ckpt_meta.get("step")
                rec["max_dim"] = int(args.max_dim)
                rec["view2_scale"] = float(args.view2_scale)
                rec["view2_contrast"] = float(args.view2_contrast)

                counts["processed"] += 1
                st = rec["status"]
                if st == "accepted":
                    counts["accepted"] += 1
                    mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    mf.flush()
                elif st == "salvaged":
                    counts["salvaged_pages"] += 1
                    counts["salvaged_systems"] += len(rec.get("salvaged_systems", []))
                    mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    mf.flush()
                else:
                    counts["rejected"] += 1
                    for r in rec.get("reject_reasons", []):
                        key = r.split("(", 1)[0]
                        reason_hist[key] = reason_hist.get(key, 0) + 1

                pf.write(img_str + "\n")
                pf.flush()

                if idx % max(1, int(args.log_every)) == 0 or st in ("accepted", "salvaged"):
                    print(
                        f"[{idx:05d}/{len(all_images):05d}] {img_path.name:36s} | "
                        f"{st.upper():8s} | lcs={rec['view_tok_lcs']:.3f} "
                        f"mtp={rec['mean_mtp_acc']:.2f} sys={v1.n_sys}/{v2.n_sys} "
                        f"tok={rec['v1_n_toks']}/{rec['v2_n_toks']} "
                        f"dec={int(v1.decode_ok)}/{int(v2.decode_ok)}",
                        flush=True,
                    )
            except Exception as exc:  # noqa: BLE001
                counts["errors"] += 1
                print(f"[{idx:05d}/{len(all_images):05d}] {img_path.name} ERROR: {exc}", flush=True)

    elapsed_total = round(time.perf_counter() - t_start, 2)
    summary = {
        "ckpt": str(args.ckpt),
        "ckpt_meta": ckpt_meta,
        "input_dir": str(input_dir),
        "out_manifest": str(out_manifest),
        "gate_config": asdict(gate_cfg),
        "counts": counts,
        "reject_reason_hist": reason_hist,
        "elapsed_secs": elapsed_total,
    }
    summary_path = out_manifest.parent / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[RISM Pseudo-Label] Done in {elapsed_total}s -> {out_manifest} (summary: {summary_path})", flush=True)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Multi-View + FSA-Gated Noisy Student RISM Pseudo-Label Generator."
    )
    ap.add_argument("--ckpt", required=True, help="Path to UniOMR teacher checkpoint (.pt).")
    ap.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help=f"Directory containing unlabeled RISM pages (default: {DEFAULT_INPUT_DIR}).",
    )
    ap.add_argument(
        "--out",
        default=str(DEFAULT_OUT_MANIFEST),
        help=f"Output resumable JSONL manifest path (default: {DEFAULT_OUT_MANIFEST}).",
    )
    ap.add_argument("--device", default="cuda", help="Inference device ('cuda' or 'cpu').")
    ap.add_argument("--bf16", action="store_true", help="Use bfloat16 autocast on CUDA.")
    ap.add_argument("--max-dim", type=int, default=1280, help="View 1 native max_dim (1280 or 1536).")
    ap.add_argument("--max-len", type=int, default=2048, help="Max generated tokens per view.")
    ap.add_argument("--view2-scale", type=float, default=0.93, help="View 2 scale factor.")
    ap.add_argument("--view2-contrast", type=float, default=0.90, help="View 2 contrast multiplier.")
    ap.add_argument("--strict-grammar", action="store_true", help="Use strict=True PrefixState.")
    ap.add_argument("--measure-beam-size", type=int, default=0, help="Optional measure-beam repair width.")
    ap.add_argument("--min-view-agree", type=float, default=0.85, help="Min full-page TokLCS F1 between View 1 & View 2.")
    ap.add_argument("--min-mtp-acc", type=float, default=0.75, help="Min mean MTP acceptance rate.")
    ap.add_argument("--min-tokens", type=int, default=24, help="Min stitched tokens per view.")
    ap.add_argument("--max-tokens", type=int, default=1950, help="Max stitched tokens per view.")
    ap.add_argument("--salvage-systems", action="store_true", help="Salvage individual matching <sys> strips.")
    ap.add_argument("--min-sys-agree", type=float, default=0.90, help="Min TokLCS F1 for salvaged <sys> strips.")
    ap.add_argument("--min-sys-tokens", type=int, default=8, help="Min tokens for a salvaged <sys> strip.")
    ap.add_argument("--resume", action="store_true", help="Resume from existing manifest + processed log.")
    ap.add_argument("--limit", type=int, default=None, help="Optional limit on number of pages to process.")
    ap.add_argument("--log-every", type=int, default=25, help="Progress log interval.")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    run_generation(args)


if __name__ == "__main__":
    main()
