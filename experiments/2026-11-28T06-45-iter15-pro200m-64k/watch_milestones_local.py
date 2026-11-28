#!/usr/bin/env python3
"""Local RTX 4080 SUPER Milestone Watcher & Dual-Benchmark Evaluator for Iter 15 (`pro200m`, 64k).

Continuously polls Hugging Face Private Dataset Hub (`gregoreesmaa/sheet-music-converter-omr`)
under `checkpoints/iter15_pro200m_64k/milestones/` for permanent EMA milestone checkpoints
(`step_XXXXXX_ema.pt` at 1k, 2k, 4k, 8k, 16k, 32k, 64k).

For each newly discovered milestone checkpoint:
  1. Downloads `step_XXXXXX_ema.pt` to `D:\\tmp\\iter15-milestones\\`.
  2. Runs `tmp-train/mvd/eval_uni_suite31.py` on the 45-piece / 90-page `suite31_hw`
     benchmark (`data-gen/score-corpus/suite31/suite31_hw_manifest.json`).
  3. Runs UniOMR full-page inference (with Pillar 3B `neg_control_gate`, FSA grammar,
     MTP-4 speculative decoding, and Rust `assembly-serialize.exe` MusicXML export) on
     the 117-page `OMR-NorthStar v0.2` pack (`D:\\tmp\\northstar-eval-pack-v0.2\\manifest.json`),
     then scores via `tmp-train/mvd/eval_northstar_runner.py`.
  4. Atomically updates the live leaderboard JSON at:
     `experiments/2026-11-28T06-45-iter15-pro200m-64k/milestones.json`.

Usage:
  # Continuous polling daemon (default 120s interval):
  .\\.venv\\Scripts\\python.exe -u experiments\\2026-11-28T06-45-iter15-pro200m-64k\\watch_milestones_local.py

  # Single pass (evaluate any pending milestones and exit):
  .\\.venv\\Scripts\\python.exe -u experiments\\2026-11-28T06-45-iter15-pro200m-64k\\watch_milestones_local.py --once
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

EXP_ID = "2026-11-28T06-45-iter15-pro200m-64k"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_MILESTONES_PREFIX = "checkpoints/iter15_pro200m_64k/milestones"

REPO_ROOT = Path(__file__).resolve().parents[2]
MVD_DIR = REPO_ROOT / "tmp-train" / "mvd"
DEFAULT_DOWNLOAD_DIR = Path(r"D:\tmp\iter15-milestones")
DEFAULT_SUITE31_MANIFEST = (
    REPO_ROOT / "data-gen" / "score-corpus" / "suite31" / "suite31_hw_manifest.json"
)
DEFAULT_NORTHSTAR_PACK = Path(r"D:\tmp\northstar-eval-pack-v0.2")
DEFAULT_NORTHSTAR_ROOT = Path(r"C:\Users\grego\projects\omr-northstar")
DEFAULT_ASER_BIN = (
    REPO_ROOT
    / "converter"
    / "assembly-serialize"
    / "target"
    / "release"
    / ("assembly-serialize.exe" if os.name == "nt" else "assembly-serialize")
)
DEFAULT_LEADERBOARD_PATH = (
    REPO_ROOT / "experiments" / EXP_ID / "milestones.json"
)

_MILESTONE_FILE_RE = re.compile(
    rf"^{re.escape(HF_MILESTONES_PREFIX)}/(step_(\d+)_ema\.pt)$"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(msg: str) -> None:
    print(f"[watch-iter15 {_utc_now()}] {msg}", flush=True)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _read_hf_token() -> Optional[str]:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok and tok.strip():
        return tok.strip()
    cache_tok = Path.home() / ".cache" / "huggingface" / "token"
    if cache_tok.is_file():
        val = cache_tok.read_text(encoding="utf-8").strip()
        if val:
            return val
    return None


def _list_hf_milestones(hf_token: Optional[str]) -> List[Tuple[int, str, str]]:
    """Return sorted list of `(step, filename, hf_repo_path)` under HF milestones/."""
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token) if hf_token else HfApi()
    files = api.list_repo_files(repo_id=HF_REPO_ID, repo_type="dataset")
    found: List[Tuple[int, str, str]] = []
    for rf in files:
        m = _MILESTONE_FILE_RE.match(rf)
        if m:
            fname = m.group(1)
            step = int(m.group(2))
            found.append((step, fname, rf))
    found.sort(key=lambda t: t[0])
    return found


def _download_milestone(
    hf_token: str,
    hf_repo_path: str,
    dst_path: Path,
) -> Path:
    from huggingface_hub import hf_hub_download

    if dst_path.is_file() and dst_path.stat().st_size > 0:
        return dst_path

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    stage_dir = dst_path.parent / f".hf_dl_stage_{os.getpid()}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    _log(f"Downloading {hf_repo_path} -> {dst_path} (staging on {stage_dir})...")
    try:
        dl = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=hf_repo_path,
            repo_type="dataset",
            token=hf_token,
            local_dir=str(stage_dir),
        )
        tmp = dst_path.with_suffix(dst_path.suffix + ".tmp")
        shutil.move(str(dl), str(tmp))
        os.replace(tmp, dst_path)
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)
    size_mb = dst_path.stat().st_size / (1024 * 1024)
    _log(f"Downloaded {dst_path.name} ({size_mb:.1f} MB)")
    return dst_path


def _extract_suite31_summary(report: Mapping[str, Any]) -> Dict[str, Any]:
    s = report.get("summary", report)

    def _f1(bucket: str) -> Optional[float]:
        b = s.get(bucket)
        if isinstance(b, Mapping):
            for k in ("note_f1_mean", "strict_note_f1", "note_f1", "mean_note_f1", "f1"):
                if isinstance(b.get(k), (int, float)):
                    return round(float(b[k]), 4)
        return None

    def _dec_ok(bucket: str) -> Optional[float]:
        b = s.get(bucket)
        if isinstance(b, Mapping):
            for k in ("decode_ok_rate", "dec_ok_rate"):
                if isinstance(b.get(k), (int, float)):
                    return round(float(b[k]), 4)
        return None

    def _first(*vals: Optional[float]) -> Optional[float]:
        for v in vals:
            if v is not None:
                return v
        return None

    return {
        "printed_clean": _f1("printed_clean"),
        "printed_photo": _f1("printed_photo"),
        "hw_clean": _f1("hw_clean"),
        "hw_photo": _f1("hw_photo"),
        "all45_clean": _first(_f1("all45_clean"), _f1("all_clean")),
        "all45_photo": _first(_f1("all45_photo"), _f1("all_photo")),
        "decode_ok_clean": _first(_dec_ok("all45_clean"), _dec_ok("all_clean")),
        "decode_ok_photo": _first(_dec_ok("all45_photo"), _dec_ok("all_photo")),
    }


def _extract_northstar_summary(report: Mapping[str, Any]) -> Dict[str, Any]:
    l1 = report.get("layer1_headline", {})
    l2 = report.get("layer2_matrix_and_deltas", {})
    l3 = report.get("layer3_component_breakdown", {})

    def _pick(d: Any, *keys: str) -> Any:
        if not isinstance(d, Mapping):
            return None
        for k in keys:
            if d.get(k) is not None:
                return d[k]
        return None

    def _pick_chain(dicts: Sequence[Any], *keys: str) -> Any:
        for d in dicts:
            val = _pick(d, *keys)
            if val is not None:
                return val
        return None

    stratum_scores = l1.get("stratum_scores", {}) if isinstance(l1, Mapping) else {}
    by_dom = l2.get("by_domain", {}) if isinstance(l2, Mapping) else {}

    def _dom(code: str) -> Optional[float]:
        if isinstance(stratum_scores, Mapping):
            val = stratum_scores.get(code)
            if isinstance(val, (int, float)):
                return round(float(val), 2)
            if isinstance(val, Mapping):
                v = _pick(val, "ns_score", "mean_ns", "score")
                if isinstance(v, (int, float)):
                    return round(float(v), 2)
        entry = by_dom.get(code) if isinstance(by_dom, Mapping) else None
        if isinstance(entry, Mapping):
            val = _pick(entry, "ns_score", "mean_ns", "score")
            return round(float(val), 2) if isinstance(val, (int, float)) else None
        if isinstance(entry, (int, float)):
            return round(float(entry), 2)
        return None

    axes = l1.get("axes_macro", {}) if isinstance(l1, Mapping) else {}
    human_util = l1.get("human_utility", {}) if isinstance(l1, Mapping) else {}
    neg = (
        l2.get("adversarial_scorecard", {}).get("negative_control", {})
        if isinstance(l2, Mapping)
        else {}
    )
    comp = l3.get("overall", l3) if isinstance(l3, Mapping) else {}

    return {
        "ns_score": _pick(l1, "ns_score", "overall_ns"),
        "tier_weighted_ns": _pick(l1, "tier_weighted_ns", "tier_w_ns"),
        "CDR": _dom("CDR"),
        "ALS": _dom("ALS"),
        "RWP": _dom("RWP"),
        "HWM": _dom("HWM"),
        "S_note": _pick_chain((axes, comp), "s_note", "S_note", "note"),
        "S_struct": _pick_chain((axes, comp), "s_struct", "S_struct", "struct"),
        "S_expr": _pick_chain((axes, comp), "s_expr", "S_expr", "expr"),
        "S_play": _pick_chain((axes, comp), "s_play", "S_play", "play"),
        "median_wFPP": _pick_chain((human_util, l1), "median_wfpp", "median_wFPP"),
        "mean_NTSR": _pick_chain((human_util, l1), "mean_ntsr", "mean_NTSR"),
        "NMHR": neg.get("nmhr") if isinstance(neg, Mapping) else None,
        "neg_gate_passed": neg.get("gate_passed") if isinstance(neg, Mapping) else None,
    }


def run_suite31_eval(
    *,
    ckpt_path: Path,
    manifest_path: Path,
    out_json: Path,
    max_dim: int = 1536,
    max_len: int = 3584,
    use_oemer_dewarp: bool = True,
) -> Dict[str, Any]:
    """Run `eval_uni_suite31.py` on `suite31_hw` (90 pages) as a clean subprocess."""
    out_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-u",
        str(MVD_DIR / "eval_uni_suite31.py"),
        "--ckpt",
        str(ckpt_path),
        "--manifest",
        str(manifest_path),
        "--max-dim",
        str(max_dim),
        "--max-len",
        str(max_len),
        "--use-grammar",
        "--use-mtp-spec",
        "--bf16",
        "--out",
        str(out_json),
    ]
    if os.name == "nt":
        cmd.append("--no-cudnn")
    if use_oemer_dewarp:
        cmd.append("--oemer-dewarp")

    _log(f"[suite31_hw] $ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
    data = json.loads(out_json.read_text(encoding="utf-8"))
    return _extract_suite31_summary(data)


def _bekern_to_musicxml(
    tokens: Sequence[str],
    *,
    aser_bin: Path,
    stem: str,
    empty_xml: str,
    dv_mod: Any,
) -> Tuple[str, bool, str]:
    if not tokens:
        return empty_xml, True, "empty-tokens"
    if aser_bin.is_file():
        with tempfile.TemporaryDirectory(prefix=f"ns_bek_{stem}_") as td:
            in_bek = Path(td) / f"{stem}.bekern"
            out_xml = Path(td) / f"{stem}.musicxml"
            in_bek.write_text(" ".join(tokens) + "\n", encoding="utf-8")
            res = subprocess.run(
                [str(aser_bin), "bekern2musicxml", "--input", str(in_bek), "--out", str(out_xml)],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if res.returncode == 0 and out_xml.is_file():
                xml_txt = out_xml.read_text(encoding="utf-8", errors="replace")
                if "<score-partwise" in xml_txt:
                    return xml_txt, True, "ok-rust"
    try:
        xml_txt = dv_mod.bekern_to_musicxml_fallback(list(tokens))
        if xml_txt and "<score-partwise" in xml_txt:
            return xml_txt, True, "ok-py-fallback"
        return empty_xml, False, "empty-py-fallback"
    except Exception as exc:
        return empty_xml, False, f"export-err:{type(exc).__name__}"


def run_northstar_eval(
    *,
    ckpt_path: Path,
    step: int,
    pack_dir: Path,
    northstar_root: Path,
    aser_bin: Path,
    run_dir: Path,
    max_dim: int = 1536,
    max_len: int = 3584,
    max_mem_tokens: int = 4608,
    use_oemer_dewarp: bool = True,
) -> Dict[str, Any]:
    """Run UniOMR inference on OMR-NorthStar v0.2 (117p) and score via `eval_northstar_runner.py`."""
    if str(MVD_DIR) not in sys.path:
        sys.path.insert(0, str(MVD_DIR))

    import torch
    from PIL import Image

    if os.name == "nt":
        torch.backends.cudnn.enabled = False

    import dual_verifier as dv  # type: ignore[import-not-found]
    import eval_northstar_runner as ns_runner  # type: ignore[import-not-found]
    import eval_uni_suite31 as eus  # type: ignore[import-not-found]
    import neg_control_gate  # type: ignore[import-not-found]
    import preprocess_dewarp  # type: ignore[import-not-found]
    import uni_omr  # type: ignore[import-not-found]

    manifest_path = pack_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pages = list(manifest["pages"])

    mxml_dir = run_dir / "musicxml"
    bekern_dir = run_dir / "bekern"
    mxml_dir.mkdir(parents=True, exist_ok=True)
    bekern_dir.mkdir(parents=True, exist_ok=True)
    timing_path = run_dir / "timing.json"
    report_path = run_dir / "report.json"

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    use_bf16 = bool(use_cuda and torch.cuda.get_device_capability(0)[0] >= 8)

    model, cfg, _meta = eus.load_uni_checkpoint(ckpt_path, device=device)
    if max_mem_tokens > 0:
        router = getattr(getattr(model, "encoder", model), "router", None)
        if router is not None and hasattr(router, "max_mem_tokens"):
            router.max_mem_tokens = int(max_mem_tokens)
        cfg.max_mem_tokens = int(max_mem_tokens)

    param_count = int(sum(p.numel() for p in model.parameters()))
    _, _, eos_id, _, _, _ = eus._vocab_spec(int(cfg.vocab_size))
    decode_args = argparse.Namespace(
        max_len=max_len,
        use_grammar=True,
        strict_grammar=False,
        use_mtp_spec=True,
    )

    latencies: Dict[str, float] = {}
    conv_statuses: Dict[str, str] = {}
    if timing_path.is_file():
        try:
            prev = json.loads(timing_path.read_text(encoding="utf-8"))
            latencies.update(prev.get("latencies_s", {}))
            conv_statuses.update(prev.get("conversion_status", {}))
        except Exception:
            pass

    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    system_name = f"uni-omr-iter15-pro200m-step{step:06d}-ema"
    _log(f"[northstar_v02] Running inference on {len(pages)} pages for {system_name}...")

    try:
        for idx, rec in enumerate(pages, start=1):
            pid = str(rec["page_id"])
            out_mxml = mxml_dir / f"{pid}.musicxml"
            out_bek = bekern_dir / f"{pid}.txt"
            if out_mxml.is_file() and out_mxml.stat().st_size > 0 and pid in latencies:
                continue

            raw_img_p = Path(str(rec["image_path"]))
            img_path = raw_img_p if raw_img_p.is_absolute() else (pack_dir / raw_img_p)
            t0 = time.perf_counter()
            try:
                with Image.open(img_path) as raw_im:
                    is_non_music, gate_info = neg_control_gate.is_non_music_page(raw_im)
                    if is_non_music:
                        _atomic_write_text(out_bek, "\n")
                        _atomic_write_text(out_mxml, ns_runner.EMPTY_VALID_MUSICXML)
                        latencies[pid] = round(time.perf_counter() - t0, 4)
                        conv_statuses[pid] = f"neg-gate:{gate_info['reason']}"
                        continue
                    work_rgb = raw_im.convert("RGB")
                    if use_oemer_dewarp:
                        work_rgb, _dw_meta = preprocess_dewarp.oemer_staffline_vertical_dewarp(work_rgb)
                    pil_im = uni_omr._resize_fullpage_pil(work_rgb, max_dim)

                img_t = uni_omr._pil_to_chw_tensor(pil_im).unsqueeze(0).to(device)
                seq, _, _, _ = eus._decode_single_view(
                    model=model,
                    cfg=cfg,
                    img_t=img_t,
                    args=decode_args,
                    device=device,
                    use_bf16=use_bf16,
                    beam_size=1,
                    beam_length_penalty=0.6,
                    measure_beam_size=0,
                    compute_logprob=False,
                )
                seq_ids = seq[0].detach().cpu().tolist()
                pred_raw_toks = eus.decode_uni_compat(
                    seq_ids, vocab_size=int(cfg.vocab_size), skip_special=True
                )
                if eos_id not in seq_ids[1:]:
                    pred_raw_toks = eus.trim_truncated_tokens(pred_raw_toks)
                hyp_page_toks, _ = eus.stitch_uni_tokens(pred_raw_toks)
                _atomic_write_text(out_bek, " ".join(hyp_page_toks) + "\n")
                xml_str, _ok, status = _bekern_to_musicxml(
                    hyp_page_toks,
                    aser_bin=aser_bin,
                    stem=pid,
                    empty_xml=ns_runner.EMPTY_VALID_MUSICXML,
                    dv_mod=dv,
                )
                _atomic_write_text(out_mxml, xml_str)
                conv_statuses[pid] = status
                latencies[pid] = round(time.perf_counter() - t0, 4)
            except Exception as exc:
                conv_statuses[pid] = f"infer-err:{type(exc).__name__}"
                _log(f"[northstar_v02 WARN] {pid} failed: {type(exc).__name__}: {exc}")
                if use_cuda:
                    torch.cuda.empty_cache()

            if idx % 20 == 0 or idx == len(pages):
                _log(f"[northstar_v02] {idx}/{len(pages)} pages decoded ({pid}, {conv_statuses[pid]})")
    finally:
        peak_mem_gb = (
            round(torch.cuda.max_memory_allocated(device) / (1024**3), 3) if use_cuda else None
        )
        del model
        gc.collect()
        if use_cuda:
            torch.cuda.empty_cache()

    _atomic_write_json(
        timing_path,
        {
            "system_name": system_name,
            "parameter_count": param_count,
            "oemer_dewarp": use_oemer_dewarp,
            "peak_memory_gb": peak_mem_gb,
            "latencies_s": latencies,
            "conversion_status": conv_statuses,
        },
    )

    # Score with official eval_northstar_runner.py CLI
    score_cmd = [
        sys.executable,
        "-u",
        str(MVD_DIR / "eval_northstar_runner.py"),
        "--manifest",
        str(manifest_path),
        "--pred-dir",
        str(mxml_dir),
        "--system-name",
        system_name,
        "--northstar-root",
        str(northstar_root),
        "--timing-json",
        str(timing_path),
        "--out",
        str(report_path),
    ]
    _log(f"[northstar_v02] $ {' '.join(score_cmd)}")
    subprocess.run(score_cmd, cwd=str(pack_dir), check=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return _extract_northstar_summary(report)


def _load_leaderboard(path: Path) -> Dict[str, Any]:
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                return loaded
        except Exception:
            pass
    return {
        "exp_id": EXP_ID,
        "hf_repo_id": HF_REPO_ID,
        "hf_milestones_prefix": HF_MILESTONES_PREFIX,
        "updated_utc": _utc_now(),
        "best": {},
        "milestones": {},
    }


def _recompute_best(lb: Dict[str, Any]) -> None:
    best_s31_clean: Optional[Tuple[float, str]] = None
    best_s31_hw: Optional[Tuple[float, str]] = None
    best_ns: Optional[Tuple[float, str]] = None
    for key, rec in lb.get("milestones", {}).items():
        if not isinstance(rec, Mapping):
            continue
        s31 = rec.get("suite31_hw") or {}
        ns = rec.get("northstar_v02") or {}
        if isinstance(s31.get("all45_clean"), (int, float)):
            v = float(s31["all45_clean"])
            if best_s31_clean is None or v > best_s31_clean[0]:
                best_s31_clean = (v, key)
        if isinstance(s31.get("hw_clean"), (int, float)):
            v = float(s31["hw_clean"])
            if best_s31_hw is None or v > best_s31_hw[0]:
                best_s31_hw = (v, key)
        if isinstance(ns.get("ns_score"), (int, float)):
            v = float(ns["ns_score"])
            if best_ns is None or v > best_ns[0]:
                best_ns = (v, key)

    lb["best"] = {
        "suite31_all45_clean": {"step": best_s31_clean[1], "value": best_s31_clean[0]}
        if best_s31_clean
        else None,
        "suite31_hw_clean": {"step": best_s31_hw[1], "value": best_s31_hw[0]}
        if best_s31_hw
        else None,
        "northstar_v02_ns": {"step": best_ns[1], "value": best_ns[0]} if best_ns else None,
    }


def poll_once(args: argparse.Namespace, hf_token: Optional[str]) -> int:
    lb = _load_leaderboard(args.leaderboard)
    milestones = _list_hf_milestones(hf_token)
    if not milestones:
        _log(f"No milestones found yet under HF {HF_MILESTONES_PREFIX}/.")
        return 0

    evaluated_now = 0
    for step, fname, hf_repo_path in milestones:
        step_key = f"step_{step:06d}"
        existing = lb.get("milestones", {}).get(step_key)
        if isinstance(existing, Mapping) and existing.get("status") == "COMPLETED":
            continue

        _log(f"=== Evaluating new milestone {step_key} ({hf_repo_path}) ===")
        ckpt_local = args.download_dir / fname
        evals_root = args.download_dir / "evals"
        s31_json = evals_root / f"{step_key}_suite31_hw.json"
        ns_dir = evals_root / f"{step_key}_northstar_v02"

        t0 = time.monotonic()
        try:
            _download_milestone(hf_token, hf_repo_path, ckpt_local)

            s31_summary = (
                run_suite31_eval(
                    ckpt_path=ckpt_local,
                    manifest_path=args.suite31_manifest,
                    out_json=s31_json,
                    max_dim=args.max_dim,
                    max_len=args.max_len,
                    use_oemer_dewarp=not args.no_oemer_dewarp,
                )
                if not args.skip_suite31
                else None
            )

            ns_summary = (
                run_northstar_eval(
                    ckpt_path=ckpt_local,
                    step=step,
                    pack_dir=args.northstar_pack,
                    northstar_root=args.northstar_root,
                    aser_bin=args.aser_bin,
                    run_dir=ns_dir,
                    max_dim=args.max_dim,
                    max_len=args.max_len,
                    use_oemer_dewarp=not args.no_oemer_dewarp,
                )
                if not args.skip_northstar
                else None
            )

            lb.setdefault("milestones", {})[step_key] = {
                "step": step,
                "hf_path": hf_repo_path,
                "local_ckpt": str(ckpt_local),
                "status": "COMPLETED",
                "evaluated_utc": _utc_now(),
                "elapsed_sec": round(time.monotonic() - t0, 2),
                "suite31_hw": s31_summary,
                "suite31_json": str(s31_json) if s31_summary is not None else None,
                "northstar_v02": ns_summary,
                "northstar_report": str(ns_dir / "report.json") if ns_summary is not None else None,
            }
            lb["updated_utc"] = _utc_now()
            _recompute_best(lb)
            # Keep milestones dict ordered by numeric step
            lb["milestones"] = dict(
                sorted(lb["milestones"].items(), key=lambda kv: int(kv[1].get("step", 0)))
            )
            _atomic_write_json(args.leaderboard, lb)
            _log(
                f"[DONE] {step_key} | suite31={s31_summary} | northstar={ns_summary} "
                f"-> updated {args.leaderboard}"
            )
            evaluated_now += 1
        except Exception as exc:
            traceback.print_exc()
            lb.setdefault("milestones", {})[step_key] = {
                "step": step,
                "hf_path": hf_repo_path,
                "local_ckpt": str(ckpt_local),
                "status": "FAILED",
                "error": f"{type(exc).__name__}: {exc}",
                "evaluated_utc": _utc_now(),
                "elapsed_sec": round(time.monotonic() - t0, 2),
            }
            lb["updated_utc"] = _utc_now()
            _atomic_write_json(args.leaderboard, lb)

    return evaluated_now


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Local milestone watcher & dual-benchmark evaluator for Iter 15 pro200m 64k."
    )
    p.add_argument("--download-dir", type=Path, default=DEFAULT_DOWNLOAD_DIR)
    p.add_argument("--suite31-manifest", type=Path, default=DEFAULT_SUITE31_MANIFEST)
    p.add_argument("--northstar-pack", type=Path, default=DEFAULT_NORTHSTAR_PACK)
    p.add_argument("--northstar-root", type=Path, default=DEFAULT_NORTHSTAR_ROOT)
    p.add_argument("--aser-bin", type=Path, default=DEFAULT_ASER_BIN)
    p.add_argument("--leaderboard", type=Path, default=DEFAULT_LEADERBOARD_PATH)
    p.add_argument("--max-dim", type=int, default=1536)
    p.add_argument("--max-len", type=int, default=3584)
    p.add_argument("--no-oemer-dewarp", action="store_true")
    p.add_argument("--skip-suite31", action="store_true")
    p.add_argument("--skip-northstar", action="store_true")
    p.add_argument("--poll-seconds", type=float, default=120.0)
    p.add_argument("--once", action="store_true", help="Run one poll pass and exit.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    hf_token = _read_hf_token()
    if not hf_token:
        raise RuntimeError(
            "HF token not found. Set HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) or login via "
            "huggingface-cli so ~/.cache/huggingface/token exists."
        )
    _log(
        f"Watching HF {HF_REPO_ID}/{HF_MILESTONES_PREFIX}/ "
        f"(download_dir={args.download_dir}, leaderboard={args.leaderboard}, once={args.once})"
    )

    while True:
        try:
            poll_once(args, hf_token)
        except Exception as exc:
            _log(f"[WARN] Poll iteration failed: {exc}")
        if args.once:
            break
        time.sleep(max(5.0, float(args.poll_seconds)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
