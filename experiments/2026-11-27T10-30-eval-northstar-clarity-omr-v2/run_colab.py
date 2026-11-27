#!/usr/bin/env python3
"""Modal L4 / Colab L4 evaluator for Clarity-OMR (`clquwu/Clarity-OMR`) on OMR-NorthStar (125 pages, v2).

Upgrades over v1 (`2026-11-26T21-50-eval-northstar-clarity-omr`):
  - Clones/updates `https://github.com/gregoreesmaa/omr-northstar.git` (`main`, >= `db12da9`) and
    prepends it to `sys.path` BEFORE importing `omr_northstar`, ensuring the newly audited `db12da9`
    scorer is used instead of the stale snapshot bundled inside `northstar_eval_pack_125p.tar.gz`.
  - Archives all 125 raw predicted `.musicxml` files into
    `evals/2026-11-27T10-30-eval-northstar-clarity-omr-v2_musicxml.tar.gz` and uploads it to HF
    Private Hub (`gregoreesmaa/sheet-music-converter-omr`) alongside the report & per-page JSONs,
    enabling 4-second CPU-only future rescores.

Pipeline:
  1. Download & unpack self-contained `datasets/northstar_eval_pack_125p.tar.gz` from HF Private Hub
     (125 pages: 31 CDR + 31 ALS + 31 RWP + 28 HWM + 4 Pillar 3B Negative Controls).
  2. Clone/update `gregoreesmaa/omr-northstar` (`main` / >= `db12da9`) for official 4-layer scoring.
  3. Clone `clquwu/Clarity-OMR` (GPL-3.0) and pre-resolve HF weights `clquwu/Clarity-OMR` (CC-BY-SA-4.0,
     ungated: `yolo.pt` Stage-A staff detector + `model.safetensors` DaViT-Base/RoPE/DoRA Stage-B).
  4. Wrap each page image as a single-page 300-DPI PDF and invoke `omr.py` (beam width 5, CUDA) ->
     `.musicxml` with atomic per-page resume and per-page timeout guard.
  5. Score all 125 pages with the fresh `omr_northstar` (`>= db12da9`) 4-layer scorer.
  6. Archive all 125 `.musicxml` predictions into a `.tar.gz` and upload
     `evals/2026-11-27T10-30-eval-northstar-clarity-omr-v2.json`, `_per_page.json`,
     `_musicxml.tar.gz`, and `runs/*.json` to `gregoreesmaa/sheet-music-converter-omr`.

Never echoes secrets (`HF_TOKEN` is read from env or `/tmp/.hf_token`).
"""

from __future__ import annotations

import gc
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

EXP_ID = "2026-11-27T10-30-eval-northstar-clarity-omr-v2"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_PACK_TAR = "datasets/northstar_eval_pack_125p.tar.gz"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_PER_PAGE_PATH = f"evals/{EXP_ID}_per_page.json"
HF_MUSICXML_TAR_PATH = f"evals/{EXP_ID}_musicxml.tar.gz"
HF_PARTIAL_TAR_PATH = f"checkpoints/{EXP_ID}/partial_musicxml.tar.gz"
HF_RUN_PATH = f"runs/{EXP_ID}.json"
PARTIAL_SYNC_EVERY_PAGES = 25

NORTHSTAR_REPO_URL = "https://github.com/gregoreesmaa/omr-northstar.git"
NORTHSTAR_LOCAL_DIR = pathlib.Path("/tmp/omr-northstar")
NORTHSTAR_GIT_REF = os.environ.get("NORTHSTAR_GIT_REF", "main")

MODEL_ID = "clquwu/Clarity-OMR"
SYSTEM_NAME = os.environ.get("NORTHSTAR_SYSTEM_NAME", "clarity-omr")
CLARITY_REPO_URL = "https://github.com/clquwu/Clarity-OMR.git"
CLARITY_LOCAL_DIR = pathlib.Path("/tmp/Clarity-OMR")
BEAM_WIDTH = int(os.environ.get("CLARITY_BEAM_WIDTH", "5"))
PDF_DPI = int(os.environ.get("CLARITY_PDF_DPI", "300"))
PAGE_TIMEOUT_SECS = int(os.environ.get("CLARITY_PAGE_TIMEOUT", "300"))
LIMIT_PAGES = int(os.environ.get("NORTHSTAR_LIMIT", "0"))  # 0 = all 125 pages

EMPTY_VALID_MUSICXML = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<score-partwise version="3.1">\n'
    '  <part-list><score-part id="P1"><part-name>Empty</part-name></score-part></part-list>\n'
    '  <part id="P1"/>\n'
    "</score-partwise>\n"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _get_hf_token() -> Optional[str]:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok.strip()
    for cand in (pathlib.Path("/tmp/.hf_token"), pathlib.Path.home() / ".cache" / "huggingface" / "token"):
        if cand.is_file():
            val = cand.read_text(encoding="utf-8").strip()
            if val:
                return val
    return None


def _get_vol_root() -> pathlib.Path:
    if pathlib.Path("/vol").is_dir():
        return pathlib.Path("/vol")
    if pathlib.Path("/content").is_dir():
        return pathlib.Path("/content/vol")
    return pathlib.Path("/tmp/omr_vol")


def _atomic_write_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_json(path: pathlib.Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _ensure_packages() -> None:
    """Install missing runtime packages using importlib.util.find_spec (no stale sys.modules)."""
    needed: List[str] = []
    for mod, pkg in (
        ("huggingface_hub", "huggingface_hub>=0.26"),
        ("fitz", "pymupdf"),
        ("ultralytics", "ultralytics"),
        ("timm", "timm"),
        ("transformers", "transformers>=4.45.0"),
        ("peft", "peft"),
        ("safetensors", "safetensors"),
        ("music21", "music21"),
        ("lxml", "lxml"),
        ("skimage", "scikit-image"),
        ("yaml", "PyYAML"),
        ("PIL", "Pillow"),
    ):
        if importlib.util.find_spec(mod) is None:
            needed.append(pkg)
    if needed:
        print(f"[setup] Installing missing packages: {needed}", flush=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--upgrade-strategy", "only-if-needed", *needed],
            check=True,
        )


def _ensure_northstar_repo() -> Tuple[pathlib.Path, str]:
    """Clone or update `gregoreesmaa/omr-northstar` (`main` / >= `db12da9`) for audited scoring."""
    if (NORTHSTAR_LOCAL_DIR / ".git").is_dir():
        print(
            f"[setup] Updating {NORTHSTAR_LOCAL_DIR} from {NORTHSTAR_REPO_URL} (ref={NORTHSTAR_GIT_REF})",
            flush=True,
        )
        subprocess.run(["git", "-C", str(NORTHSTAR_LOCAL_DIR), "fetch", "origin"], check=True)
        res = subprocess.run(
            ["git", "-C", str(NORTHSTAR_LOCAL_DIR), "checkout", "--force", f"origin/{NORTHSTAR_GIT_REF}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if res.returncode != 0:
            subprocess.run(
                ["git", "-C", str(NORTHSTAR_LOCAL_DIR), "checkout", "--force", NORTHSTAR_GIT_REF],
                check=True,
            )
    else:
        print(f"[setup] Cloning {NORTHSTAR_REPO_URL} -> {NORTHSTAR_LOCAL_DIR}", flush=True)
        if NORTHSTAR_LOCAL_DIR.exists():
            shutil.rmtree(NORTHSTAR_LOCAL_DIR, ignore_errors=True)
        subprocess.run(["git", "clone", NORTHSTAR_REPO_URL, str(NORTHSTAR_LOCAL_DIR)], check=True)
        if NORTHSTAR_GIT_REF and NORTHSTAR_GIT_REF != "main":
            subprocess.run(
                ["git", "-C", str(NORTHSTAR_LOCAL_DIR), "checkout", "--force", NORTHSTAR_GIT_REF],
                check=True,
            )

    if not (NORTHSTAR_LOCAL_DIR / "omr_northstar" / "scoring.py").is_file():
        raise FileNotFoundError(f"Missing omr_northstar/scoring.py in {NORTHSTAR_LOCAL_DIR}")

    sha = subprocess.check_output(
        ["git", "-C", str(NORTHSTAR_LOCAL_DIR), "rev-parse", "--short", "HEAD"],
        text=True,
    ).strip()
    repo_str = str(NORTHSTAR_LOCAL_DIR)
    if repo_str in sys.path:
        sys.path.remove(repo_str)
    sys.path.insert(0, repo_str)
    print(f"[setup] omr-northstar scorer ready at {NORTHSTAR_LOCAL_DIR} (commit={sha})", flush=True)
    return NORTHSTAR_LOCAL_DIR, sha


def _ensure_clarity_repo() -> pathlib.Path:
    if not (CLARITY_LOCAL_DIR / "omr.py").is_file():
        print(f"[setup] Cloning {CLARITY_REPO_URL} -> {CLARITY_LOCAL_DIR}", flush=True)
        if CLARITY_LOCAL_DIR.exists():
            shutil.rmtree(CLARITY_LOCAL_DIR, ignore_errors=True)
        subprocess.run(["git", "clone", "--depth", "1", CLARITY_REPO_URL, str(CLARITY_LOCAL_DIR)], check=True)
    repo_str = str(CLARITY_LOCAL_DIR)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    return CLARITY_LOCAL_DIR


def _ensure_northstar_pack(vol_root: pathlib.Path, hf_token: Optional[str]) -> pathlib.Path:
    """Download and extract `datasets/northstar_eval_pack_125p.tar.gz` into `vol_root/datasets/`."""
    from huggingface_hub import hf_hub_download

    pack_dir = vol_root / "datasets" / "northstar_eval_pack_125p"
    manifest_file = pack_dir / "manifest.json"
    if manifest_file.is_file():
        print(f"[data] Using cached NorthStar 125p pack at {pack_dir}", flush=True)
        return pack_dir

    dl_dir = vol_root / "datasets"
    dl_dir.mkdir(parents=True, exist_ok=True)
    print(f"[data] Downloading {HF_PACK_TAR} from {HF_REPO_ID} ...", flush=True)
    tar_path = pathlib.Path(
        hf_hub_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            filename=HF_PACK_TAR,
            token=hf_token,
        )
    )
    print(f"[data] Extracting {tar_path} -> {dl_dir} ...", flush=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(path=dl_dir)
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Expected {manifest_file} after extracting {tar_path}")
    print(f"[data] NorthStar pack ready at {pack_dir}", flush=True)
    return pack_dir


def _image_to_single_page_pdf(img_path: pathlib.Path, pdf_path: pathlib.Path, dpi: int = 300) -> None:
    """Wrap a page image into a single-page PDF sized so rendering at `dpi` preserves native pixels."""
    import fitz  # type: ignore[import-not-found]
    from PIL import Image

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(img_path) as im:
        w_px, h_px = im.size
    w_pt = max(72.0, w_px * 72.0 / float(dpi))
    h_pt = max(72.0, h_px * 72.0 / float(dpi))

    doc = fitz.open()
    page = doc.new_page(width=w_pt, height=h_pt)
    page.insert_image(fitz.Rect(0, 0, w_pt, h_pt), filename=str(img_path))
    tmp_pdf = pdf_path.with_suffix(".pdf.tmp")
    doc.save(str(tmp_pdf))
    doc.close()
    os.replace(tmp_pdf, pdf_path)


def _archive_musicxml(mxml_dir: pathlib.Path, out_tar: pathlib.Path, *, timing_path: Optional[pathlib.Path] = None) -> int:
    """Archive all predicted `.musicxml` files (and optional timing.json) atomically into a `.tar.gz`."""
    out_tar.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(mxml_dir.glob("*.musicxml"))
    tmp_tar = out_tar.with_suffix(out_tar.suffix + ".tmp")
    with tarfile.open(tmp_tar, "w:gz") as tf:
        for p in files:
            tf.add(p, arcname=p.name)
        if timing_path is not None and timing_path.is_file():
            tf.add(timing_path, arcname="timing.json")
    os.replace(tmp_tar, out_tar)
    print(f"[archive] Archived {len(files)} MusicXML files -> {out_tar} ({out_tar.stat().st_size} bytes)", flush=True)
    return len(files)


def _restore_partial_from_hf(
    *,
    hf_token: Optional[str],
    mxml_dir: pathlib.Path,
    timing_path: pathlib.Path,
) -> int:
    """Best-effort download and unpack of `checkpoints/<EXP_ID>/partial_musicxml.tar.gz` from HF."""
    if not hf_token:
        return 0
    try:
        from huggingface_hub import hf_hub_download

        dl_path = pathlib.Path(
            hf_hub_download(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                filename=HF_PARTIAL_TAR_PATH,
                token=hf_token,
            )
        )
        restored = 0
        with tarfile.open(dl_path, "r:gz") as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                name = pathlib.Path(member.name).name
                if name == "timing.json":
                    if not timing_path.is_file():
                        f = tf.extractfile(member)
                        if f is not None:
                            _atomic_write_text(timing_path, f.read().decode("utf-8", errors="replace"))
                elif name.endswith(".musicxml"):
                    dest = mxml_dir / name
                    if not dest.is_file():
                        f = tf.extractfile(member)
                        if f is not None:
                            tmp = dest.with_suffix(dest.suffix + ".tmp")
                            tmp.write_bytes(f.read())
                            os.replace(tmp, dest)
                            restored += 1
        print(f"[resume] Restored {restored} MusicXML files from HF {HF_PARTIAL_TAR_PATH}", flush=True)
        return restored
    except Exception as exc:
        print(f"[resume] No partial HF checkpoint restored ({type(exc).__name__}: {str(exc)[:120]})", flush=True)
        return 0


def _sync_partial_to_hf(
    *,
    hf_token: Optional[str],
    mxml_dir: pathlib.Path,
    timing_path: pathlib.Path,
    partial_tar_path: pathlib.Path,
    n_done: int,
) -> None:
    """Best-effort atomic archive + upload of incremental MusicXML predictions to HF."""
    if not hf_token or LIMIT_PAGES != 0:
        return
    try:
        from huggingface_hub import HfApi

        n_files = _archive_musicxml(mxml_dir, partial_tar_path, timing_path=timing_path)
        api = HfApi(token=hf_token)
        api.upload_file(
            path_or_fileobj=str(partial_tar_path),
            path_in_repo=HF_PARTIAL_TAR_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"ckpt({EXP_ID}): partial MusicXML checkpoint ({n_files} files, {n_done} pages)",
        )
        print(f"[hf] Synced partial checkpoint {HF_PARTIAL_TAR_PATH} ({n_files} files)", flush=True)
    except Exception as exc:
        print(f"[hf] WARN: partial checkpoint sync failed ({type(exc).__name__}: {str(exc)[:120]})", flush=True)


def _score_northstar(
    *,
    scorer_dir: pathlib.Path,
    pack_dir: pathlib.Path,
    pred_dir: pathlib.Path,
    timing_by_page: Mapping[str, float],
    telemetry: Mapping[str, Any],
    limit_pages: int = 0,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], str]:
    """Score predictions using the cloned `omr_northstar` (>= `db12da9`) stdlib scorer."""
    scorer_str = str(scorer_dir)
    if scorer_str in sys.path:
        sys.path.remove(scorer_str)
    sys.path.insert(0, scorer_str)
    from omr_northstar import cer, cli, scoring, stress, validate  # type: ignore[import-not-found]

    scoring_file = pathlib.Path(scoring.__file__).resolve()
    if not scoring_file.is_relative_to(scorer_dir.resolve()):
        raise RuntimeError(f"Expected omr_northstar from {scorer_dir}, got {scoring_file}")
    print(f"[score] Using omr_northstar scorer from {scoring_file}", flush=True)

    manifest_path = pack_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Backfill legacy pack metadata fields required by omr-northstar >= db12da9 validator
    # (datasets/northstar_eval_pack_125p.tar.gz has None for optics_codec.codec and provenance.source_type).
    for rec in manifest.get("pages", []):
        if isinstance(rec, dict):
            opt = rec.get("optics_codec")
            if isinstance(opt, dict) and not (isinstance(opt.get("codec"), str) and opt["codec"].strip()):
                img_ext = pathlib.Path(str(rec.get("image_path", ""))).suffix.lstrip(".").lower()
                opt["codec"] = img_ext or "png"
            prov = rec.get("provenance")
            if isinstance(prov, dict) and not (isinstance(prov.get("source_type"), str) and prov["source_type"].strip()):
                prov["source_type"] = "northstar_eval_pack_125p"
    m_errs = validate.validate_manifest(manifest)
    if m_errs:
        print(f"[score] WARN: Manifest validation reported {len(m_errs)} legacy schema warnings: {m_errs[:5]}", flush=True)

    pages_spec: Sequence[Mapping[str, Any]] = manifest["pages"]
    if limit_pages > 0:
        pages_spec = pages_spec[:limit_pages]

    page_metrics_list = []
    per_page_rows: List[Dict[str, Any]] = []
    neg_results: List[Dict[str, Any]] = []
    missing_preds = 0

    for rec in pages_spec:
        pid = str(rec["page_id"])
        wid = str(rec["work_id"])
        dom = str(rec["domain"])
        tier = str(rec["complexity_tier"])
        anchor = rec.get("paired_anchor_id")
        is_neg = bool(rec.get("is_negative_control", False))
        neg_type = str(rec.get("negative_control_type", "non_music"))
        prov = rec.get("provenance") if isinstance(rec.get("provenance"), Mapping) else {}
        engraver = prov.get("engraver_or_publisher") or rec.get("engraver")

        gt_path = pack_dir / str(rec["gt_musicxml"])
        gt_xml = gt_path.read_text(encoding="utf-8", errors="replace")

        pred_file = pred_dir / f"{pid}.musicxml"
        if pred_file.is_file():
            pred_source = str(pred_file.name)
            pred_xml = pred_file.read_text(encoding="utf-8", errors="replace")
        else:
            missing_preds += 1
            pred_source = "missing"
            pred_xml = ""

        lat = timing_by_page.get(pid)
        pm = cer.score_musicxml_pair(
            pred_xml,
            gt_xml,
            page_id=pid,
            work_id=wid,
            domain=dom,
            tier=tier,
            paired_anchor_id=str(anchor) if anchor else None,
            engraver=str(engraver) if engraver else None,
            latency_s=lat,
        )
        page_metrics_list.append(pm)
        pq = scoring.compute_page_quality(pm)

        row: Dict[str, Any] = {
            "page_id": pid,
            "work_id": wid,
            "domain": dom,
            "tier": tier,
            "is_negative_control": is_neg,
            "pred_source": pred_source,
            "parseable": pm.parseable,
            "q_score": round(pq.q_score, 4),
            "s_note": round(pm.s_note, 4),
            "s_struct": round(pm.s_struct, 4),
            "s_expr": round(pm.s_expr, 4),
            "s_play": round(pm.s_play, 4),
            "wfpp": round(pq.effective_wfpp, 4),
            "ntsr": round(pq.ntsr, 4),
            "n_gt": pm.n_gt,
            "n_pred": pm.n_pred,
            "m_gt": pm.m_gt,
            "m_pred": pm.m_pred,
            "bsvr": round(pm.bsvr, 4),
            "catastrophic_failure": pq.catastrophic_failure,
            "cf_reasons": list(pq.cf_reasons),
            "latency_s": lat,
        }
        if is_neg:
            nc = stress.score_negative_control(pred_note_count=pm.n_pred, page_type=neg_type)
            row["negative_control_passed"] = nc.passed
            row["negative_control_failures"] = list(nc.gate_failures)
            neg_results.append(
                {
                    "page_id": pid,
                    "page_type": neg_type,
                    "pred_note_count": nc.pred_note_count,
                    "passed": nc.passed,
                    "gate_failures": list(nc.gate_failures),
                }
            )
        per_page_rows.append(row)

    adversarial: Dict[str, Any] = {}
    if neg_results:
        n_neg = len(neg_results)
        n_pass = sum(1 for r in neg_results if r["passed"])
        n_fail = n_neg - n_pass
        adversarial["negative_control"] = {
            "n_pages": n_neg,
            "passed_count": n_pass,
            "nmhr": round(n_fail / n_neg, 4),
            "pass_rate": round(n_pass / n_neg, 4),
            "gate_passed": n_fail == 0,
            "pages": neg_results,
        }
    if missing_preds > 0:
        adversarial["missing_prediction_files"] = missing_preds

    report = scoring.build_report(
        page_metrics_list,
        system_name=SYSTEM_NAME,
        split=str(manifest.get("split", "dev-v1")),
        benchmark_version=str(manifest.get("benchmark_version", "1.0.0")),
        adversarial=adversarial if adversarial else None,
        telemetry=dict(telemetry) if telemetry else None,
        n_boot=1000,
        bootstrap_seed=42,
    )
    if limit_pages == 0:
        r_errs = validate.validate_report(report)
        if r_errs:
            raise ValueError(f"Report validation failed ({len(r_errs)} errors): {r_errs[:5]}")

    scorecard = cli.format_scorecard(report)
    return report, per_page_rows, scorecard


def main() -> int:
    t_start = time.monotonic()
    started_at = _utc_now()
    print(f"[{started_at}] Starting {EXP_ID} | model={MODEL_ID} | beam={BEAM_WIDTH}", flush=True)

    _ensure_packages()
    northstar_dir, northstar_sha = _ensure_northstar_repo()
    clarity_dir = _ensure_clarity_repo()

    import torch
    from huggingface_hub import HfApi

    hf_token = _get_hf_token()
    vol_root = _get_vol_root()
    pack_dir = _ensure_northstar_pack(vol_root, hf_token)

    # Pre-download Stage-A & Stage-B weights once so per-page runs use local cache.
    from src.model_assets import (  # type: ignore[import-not-found]
        ensure_default_stage_a_weights,
        ensure_default_stage_b_checkpoint,
    )

    stage_a_path = ensure_default_stage_a_weights(clarity_dir)
    stage_b_path = ensure_default_stage_b_checkpoint(clarity_dir)
    print(f"[model] Clarity-OMR Stage-A={stage_a_path} | Stage-B={stage_b_path}", flush=True)

    run_dir = vol_root / "runs" / EXP_ID
    mxml_dir = run_dir / "musicxml"
    pdf_dir = pathlib.Path("/tmp") / f"{EXP_ID}_pdfs"
    work_root = pathlib.Path("/tmp") / f"{EXP_ID}_work"
    mxml_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)
    timing_path = run_dir / "timing.json"
    partial_tar_path = run_dir / "partial_musicxml.tar.gz"

    _restore_partial_from_hf(hf_token=hf_token, mxml_dir=mxml_dir, timing_path=timing_path)

    latencies: Dict[str, float] = {}
    conv_statuses: Dict[str, str] = {}
    if timing_path.is_file():
        try:
            prev = json.loads(timing_path.read_text(encoding="utf-8"))
            latencies.update(prev.get("latencies_s", {}))
            conv_statuses.update(prev.get("conversion_status", {}))
        except Exception:
            pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    print(f"[hw] device={device} gpu={gpu_name}", flush=True)

    manifest = json.loads((pack_dir / "manifest.json").read_text(encoding="utf-8"))
    pages = list(manifest["pages"])
    if LIMIT_PAGES > 0:
        pages = pages[:LIMIT_PAGES]

    n_ok = 0
    n_fail = 0
    newly_completed = 0
    for idx, rec in enumerate(pages, start=1):
        pid = str(rec["page_id"])
        out_mxml = mxml_dir / f"{pid}.musicxml"

        if out_mxml.is_file() and (pid in latencies or out_mxml.stat().st_size > 0):
            status = conv_statuses.get(pid, "resumed")
            if status in ("ok", "no-staves-empty", "resumed"):
                n_ok += 1
            else:
                n_fail += 1
            continue

        img_path = pack_dir / str(rec["image_path"])
        page_pdf = pdf_dir / f"{pid}.pdf"
        page_work = work_root / pid
        tmp_mxml = out_mxml.with_suffix(".musicxml.tmp")
        if page_work.exists():
            shutil.rmtree(page_work, ignore_errors=True)
        if tmp_mxml.exists():
            tmp_mxml.unlink()

        t0 = time.perf_counter()
        ok = False
        status = "unknown"
        try:
            _image_to_single_page_pdf(img_path, page_pdf, dpi=PDF_DPI)
            cmd = [
                sys.executable,
                str(clarity_dir / "omr.py"),
                str(page_pdf),
                "-o",
                str(tmp_mxml),
                "--beam-width",
                str(BEAM_WIDTH),
                "--device",
                device,
                "--pdf-dpi",
                str(PDF_DPI),
                "--work-dir",
                str(page_work),
            ]
            proc = subprocess.run(
                cmd,
                cwd=str(clarity_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=PAGE_TIMEOUT_SECS,
                check=False,
            )
            if proc.returncode == 0 and tmp_mxml.is_file() and tmp_mxml.stat().st_size > 0:
                os.replace(tmp_mxml, out_mxml)
                ok = True
                status = "ok"
            else:
                tail = (proc.stdout or "").strip().splitlines()[-3:]
                tail_str = " | ".join(tail)[:140].replace("\n", " ")
                # If Stage-A found zero staves on a negative control or blank page, record faithfully.
                if "no staff" in tail_str.lower() or "0 staff" in tail_str.lower() or "0 staves" in tail_str.lower():
                    _atomic_write_text(out_mxml, EMPTY_VALID_MUSICXML)
                    ok = True
                    status = "no-staves-empty"
                else:
                    _atomic_write_text(out_mxml, "")
                    ok = False
                    status = f"rc={proc.returncode}({tail_str})"
        except subprocess.TimeoutExpired:
            _atomic_write_text(out_mxml, "")
            ok = False
            status = f"timeout({PAGE_TIMEOUT_SECS}s)"
        except Exception as exc:
            _atomic_write_text(out_mxml, "")
            ok = False
            status = f"infer-err({type(exc).__name__}:{str(exc)[:100]})"
        finally:
            if page_work.exists():
                shutil.rmtree(page_work, ignore_errors=True)
            if page_pdf.exists():
                page_pdf.unlink(missing_ok=True)
            if tmp_mxml.exists():
                tmp_mxml.unlink(missing_ok=True)

        elapsed = time.perf_counter() - t0
        latencies[pid] = round(elapsed, 4)
        conv_statuses[pid] = status
        if ok:
            n_ok += 1
        else:
            n_fail += 1

        _atomic_write_json(
            timing_path,
            {
                "system_name": SYSTEM_NAME,
                "model_id": MODEL_ID,
                "beam_width": BEAM_WIDTH,
                "latencies_s": latencies,
                "conversion_status": conv_statuses,
            },
        )
        print(
            f"[{idx:03d}/{len(pages):03d}] {pid:42s} ok={str(ok):5s} ({status[:60]}) {elapsed:.2f}s",
            flush=True,
        )
        newly_completed += 1
        if newly_completed % PARTIAL_SYNC_EVERY_PAGES == 0:
            _sync_partial_to_hf(
                hf_token=hf_token,
                mxml_dir=mxml_dir,
                timing_path=timing_path,
                partial_tar_path=partial_tar_path,
                n_done=n_ok + n_fail,
            )

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    telemetry = {
        "peak_memory_gb": 0.0,
        "parameter_count": 0,
        "beam_width": BEAM_WIDTH,
        "pdf_dpi": PDF_DPI,
    }

    report_path = run_dir / "report.json"
    per_page_path = run_dir / "per_page.json"
    musicxml_tar_path = run_dir / f"{EXP_ID}_musicxml.tar.gz"
    run_ledger_path = run_dir / "run_ledger.json"

    # Archive and upload raw MusicXML predictions BEFORE scoring so 70-min GPU inference is never lost.
    n_archived = _archive_musicxml(mxml_dir, musicxml_tar_path)
    musicxml_uploaded = False
    if hf_token and LIMIT_PAGES == 0:
        api = HfApi(token=hf_token)
        api.upload_file(
            path_or_fileobj=str(musicxml_tar_path),
            path_in_repo=HF_MUSICXML_TAR_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"eval: {EXP_ID} NorthStar 125p raw MusicXML archive",
        )
        musicxml_uploaded = True
        print(f"[hf] Uploaded {HF_MUSICXML_TAR_PATH} (pre-score checkpoint)", flush=True)

    report, per_page_rows, scorecard = _score_northstar(
        scorer_dir=northstar_dir,
        pack_dir=pack_dir,
        pred_dir=mxml_dir,
        timing_by_page=latencies,
        telemetry=telemetry,
        limit_pages=LIMIT_PAGES,
    )

    _atomic_write_json(report_path, report)
    _atomic_write_json(
        per_page_path,
        {
            "exp_id": EXP_ID,
            "system_name": SYSTEM_NAME,
            "model_id": MODEL_ID,
            "northstar_scorer_commit": northstar_sha,
            "pages": per_page_rows,
        },
    )
    _atomic_write_json(
        run_ledger_path,
        {
            "exp_id": EXP_ID,
            "system_name": SYSTEM_NAME,
            "model_id": MODEL_ID,
            "northstar_scorer_commit": northstar_sha,
            "gpu": gpu_name,
            "started_at": started_at,
            "completed_at": _utc_now(),
            "elapsed_sec": round(time.monotonic() - t_start, 2),
            "pages_total": len(pages),
            "conversion_ok": n_ok,
            "conversion_fail": n_fail,
            "musicxml_archived_count": n_archived,
            "telemetry": telemetry,
            "headline": report.get("layer1_headline", {}),
            "hf_eval_path": HF_EVAL_PATH,
            "hf_per_page_path": HF_PER_PAGE_PATH,
            "hf_musicxml_tar_path": HF_MUSICXML_TAR_PATH,
        },
    )

    print("\n" + scorecard + "\n", flush=True)

    if hf_token and LIMIT_PAGES == 0:
        api = HfApi(token=hf_token)
        upload_items = [
            (report_path, HF_EVAL_PATH, f"eval: {EXP_ID} NorthStar 125p report"),
            (per_page_path, HF_PER_PAGE_PATH, f"eval: {EXP_ID} NorthStar 125p per-page"),
            (run_ledger_path, HF_RUN_PATH, f"run: {EXP_ID} ledger"),
        ]
        if not musicxml_uploaded:
            upload_items.insert(2, (musicxml_tar_path, HF_MUSICXML_TAR_PATH, f"eval: {EXP_ID} NorthStar 125p raw MusicXML archive"))
        for local_p, repo_p, msg in upload_items:
            api.upload_file(
                path_or_fileobj=str(local_p),
                path_in_repo=repo_p,
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                commit_message=msg,
            )
            print(f"[hf] Uploaded {repo_p}", flush=True)
    else:
        print(f"[hf] Skipping upload (hf_token={'set' if hf_token else 'unset'}, limit={LIMIT_PAGES})", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
