#!/usr/bin/env python3
"""Modal L4 / Colab L4 evaluator for SMT++ (`PRAIG/smt-fp-grandstaff`) on OMR-NorthStar (125 pages).

Pipeline:
  1. Download & unpack self-contained `datasets/northstar_eval_pack_125p.tar.gz` from HF Private Hub
     (125 pages: 31 CDR + 31 ALS + 31 RWP + 28 HWM + 4 Pillar 3B Negative Controls + `omr_northstar`).
  2. Clone `antoniorv6/SMT` and load `PRAIG/smt-fp-grandstaff` (`SMTModelForCausalLM`) on CUDA L4.
  3. Run full-page inference -> Humdrum `**kern` -> `music21` -> `.musicxml` (atomic per-page resume).
  4. Score all 125 pages with the official `omr_northstar` 4-layer scorer.
  5. Upload `evals/2026-11-26T15-45-eval-northstar-smtpp.json` (plus `_per_page.json` and `runs/*.json`)
     to `gregoreesmaa/sheet-music-converter-omr`.

Never echoes secrets (`HF_TOKEN` is read from env or `/tmp/.hf_token`).
"""

from __future__ import annotations

import gc
import json
import os
import pathlib
import subprocess
import sys
import tarfile
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

EXP_ID = "2026-11-26T15-45-eval-northstar-smtpp"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_PACK_TAR = "datasets/northstar_eval_pack_125p.tar.gz"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_PER_PAGE_PATH = f"evals/{EXP_ID}_per_page.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

MODEL_ID = os.environ.get("SMT_MODEL_ID", "PRAIG/smt-fp-grandstaff")
SYSTEM_NAME = os.environ.get("NORTHSTAR_SYSTEM_NAME", "smtpp-fp-grandstaff")
SMT_REPO_URL = "https://github.com/antoniorv6/SMT.git"
SMT_LOCAL_DIR = pathlib.Path("/tmp/SMT")

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
    """Install lightweight missing runtime packages without overwriting PyTorch."""
    needed: List[str] = []
    for mod, pkg in (
        ("transformers", "transformers>=4.45.0,<5.0.0"),
        ("music21", "music21"),
        ("cv2", "opencv-python-headless"),
        ("huggingface_hub", "huggingface_hub>=0.26"),
        ("safetensors", "safetensors"),
    ):
        try:
            __import__(mod)
        except ImportError:
            needed.append(pkg)
    if needed:
        print(f"[setup] Installing missing packages: {needed}", flush=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--no-deps" if False else "--upgrade-strategy", "only-if-needed", *needed],
            check=True,
        )


def _ensure_smt_repo() -> pathlib.Path:
    if not (SMT_LOCAL_DIR / "smt_model").is_dir():
        print(f"[setup] Cloning {SMT_REPO_URL} -> {SMT_LOCAL_DIR}", flush=True)
        if SMT_LOCAL_DIR.exists():
            subprocess.run(["rm", "-rf", str(SMT_LOCAL_DIR)], check=False)
        subprocess.run(["git", "clone", "--depth", "1", SMT_REPO_URL, str(SMT_LOCAL_DIR)], check=True)
    smt_str = str(SMT_LOCAL_DIR)
    if smt_str not in sys.path:
        sys.path.insert(0, smt_str)
    return SMT_LOCAL_DIR


def _ensure_northstar_pack(vol_root: pathlib.Path, hf_token: Optional[str]) -> pathlib.Path:
    """Download and extract `datasets/northstar_eval_pack_125p.tar.gz` into `vol_root/datasets/`."""
    from huggingface_hub import hf_hub_download

    pack_dir = vol_root / "datasets" / "northstar_eval_pack_125p"
    manifest_file = pack_dir / "manifest.json"
    if manifest_file.is_file() and (pack_dir / "omr_northstar" / "scoring.py").is_file():
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


def _load_smt_model(model_id: str, device: str):
    """Load `SMTModelForCausalLM` cleanly, disabling flash_attn if unavailable."""
    import torch
    from smt_model import SMTConfig, SMTModelForCausalLM  # type: ignore[import-not-found]

    has_flash = False
    try:
        import flash_attn  # noqa: F401
        has_flash = True
    except ImportError:
        has_flash = False

    cfg = SMTConfig.from_pretrained(model_id)
    if getattr(cfg, "use_flash_attn", False) and not has_flash:
        print("[model] flash_attn not installed; setting config.use_flash_attn = False", flush=True)
        cfg.use_flash_attn = False

    model = SMTModelForCausalLM.from_pretrained(model_id, config=cfg)
    model = model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[model] Loaded {model_id} on {device} | params={n_params/1e6:.2f}M | "
        f"maxh={getattr(cfg, 'maxh', '?')} maxw={getattr(cfg, 'maxw', '?')} maxlen={getattr(cfg, 'maxlen', '?')}",
        flush=True,
    )
    return model, cfg, n_params


def _preprocess_smt_image(img_path: pathlib.Path, maxh: int, maxw: int):
    """Load image, resize within (maxh, maxw) preserving aspect ratio, convert to SMT tensor [1, 1, H, W]."""
    import cv2
    import numpy as np
    import torch

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cv2.imread failed for {img_path}")

    h, w = img.shape[:2]
    scale = min(1.0, maxh / max(h, 1), maxw / max(w, 1))
    if scale < 1.0:
        new_w = max(32, int(round(w * scale)))
        new_h = max(32, int(round(h * scale)))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Prefer official SMT convert_img_to_tensor if available; fallback to exact grayscale ToTensor.
    try:
        from data_augmentation.data_augmentation import convert_img_to_tensor  # type: ignore[import-not-found]

        tensor = convert_img_to_tensor(img)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        return tensor
    except Exception:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        arr = gray.astype(np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def _format_smt_kern(raw_preds: Any) -> str:
    """Decode SMT++ token sequence into Humdrum **kern text."""
    if isinstance(raw_preds, (list, tuple)):
        if raw_preds and isinstance(raw_preds[0], (list, tuple)):
            tokens = [str(t) for t in raw_preds[0]]
        else:
            tokens = [str(t) for t in raw_preds]
        text = "".join(tokens)
    else:
        text = str(raw_preds)

    text = (
        text.replace("<bos>", "")
        .replace("<eos>", "")
        .replace("<pad>", "")
        .replace("<b>", "\n")
        .replace("<s>", " ")
        .replace("<t>", "\t")
        .strip()
    )
    if not text:
        return ""

    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""

    # Determine spine count from first non-comment line and ensure valid **kern header/footer.
    first_cols = max(1, len(lines[0].split("\t")))
    if not lines[0].startswith("**"):
        lines.insert(0, "\t".join(["**kern"] * first_cols))
    if not lines[-1].startswith("*-"):
        last_cols = max(1, len(lines[-1].split("\t")))
        lines.append("\t".join(["*-"] * last_cols))
    return "\n".join(lines) + "\n"


def _kern_to_musicxml(kern_text: str, out_mxml_path: pathlib.Path) -> Tuple[bool, str]:
    """Convert Humdrum **kern text to MusicXML via music21."""
    if not kern_text.strip():
        _atomic_write_text(out_mxml_path, "")
        return False, "empty-kern"

    try:
        import music21

        score = music21.converter.parseData(kern_text, format="humdrum")
        tmp_path = out_mxml_path.with_suffix(".musicxml.tmp")
        out_mxml_path.parent.mkdir(parents=True, exist_ok=True)
        score.write("musicxml", fp=str(tmp_path))
        os.replace(tmp_path, out_mxml_path)
        return True, "ok"
    except Exception as exc:
        _atomic_write_text(out_mxml_path, "")
        msg = f"{type(exc).__name__}: {str(exc)[:140]}".replace("\n", " ")
        return False, f"m21-fail({msg})"


def _score_northstar(
    *,
    pack_dir: pathlib.Path,
    pred_dir: pathlib.Path,
    timing_by_page: Mapping[str, float],
    telemetry: Mapping[str, Any],
    limit_pages: int = 0,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], str]:
    """Score predictions using the bundled `omr_northstar` stdlib scorer."""
    pack_str = str(pack_dir)
    if pack_str not in sys.path:
        sys.path.insert(0, pack_str)
    from omr_northstar import cer, cli, scoring, stress, validate  # type: ignore[import-not-found]

    manifest_path = pack_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    m_errs = validate.validate_manifest(manifest)
    if m_errs:
        raise ValueError(f"Manifest validation failed ({len(m_errs)} errors): {m_errs[:5]}")

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
    print(f"[{started_at}] Starting {EXP_ID} | model={MODEL_ID}", flush=True)

    _ensure_packages()
    _ensure_smt_repo()

    import torch
    from huggingface_hub import HfApi

    hf_token = _get_hf_token()
    vol_root = _get_vol_root()
    pack_dir = _ensure_northstar_pack(vol_root, hf_token)

    run_dir = vol_root / "runs" / EXP_ID
    mxml_dir = run_dir / "musicxml"
    kern_dir = run_dir / "kern"
    mxml_dir.mkdir(parents=True, exist_ok=True)
    kern_dir.mkdir(parents=True, exist_ok=True)
    timing_path = run_dir / "timing.json"

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

    model, cfg, n_params = _load_smt_model(MODEL_ID, device)
    maxh = int(getattr(cfg, "maxh", 2970))
    maxw = int(getattr(cfg, "maxw", 2100))

    manifest = json.loads((pack_dir / "manifest.json").read_text(encoding="utf-8"))
    pages = list(manifest["pages"])
    if LIMIT_PAGES > 0:
        pages = pages[:LIMIT_PAGES]

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    n_ok = 0
    n_fail = 0
    for idx, rec in enumerate(pages, start=1):
        pid = str(rec["page_id"])
        out_mxml = mxml_dir / f"{pid}.musicxml"
        out_kern = kern_dir / f"{pid}.krn"

        if out_mxml.is_file() and pid in latencies:
            status = conv_statuses.get(pid, "resumed")
            if status == "ok":
                n_ok += 1
            else:
                n_fail += 1
            continue

        img_path = pack_dir / str(rec["image_path"])
        t0 = time.perf_counter()
        try:
            tensor_img = _preprocess_smt_image(img_path, maxh=maxh, maxw=maxw).to(device)
            with torch.inference_mode():
                preds, _ = model.predict(tensor_img, convert_to_str=True)
            kern_text = _format_smt_kern(preds)
            _atomic_write_text(out_kern, kern_text)
            ok, status = _kern_to_musicxml(kern_text, out_mxml)
        except Exception as exc:
            _atomic_write_text(out_kern, f"! ERROR: {type(exc).__name__}: {exc}\n")
            _atomic_write_text(out_mxml, "")
            ok = False
            status = f"infer-err({type(exc).__name__}:{str(exc)[:100]})"
            if device == "cuda":
                torch.cuda.empty_cache()

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
                "latencies_s": latencies,
                "conversion_status": conv_statuses,
            },
        )
        print(
            f"[{idx:03d}/{len(pages):03d}] {pid:42s} ok={str(ok):5s} ({status[:60]}) {elapsed:.2f}s",
            flush=True,
        )

    peak_mem_gb = round(torch.cuda.max_memory_allocated() / 1e9, 3) if device == "cuda" else 0.0
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    telemetry = {
        "peak_memory_gb": peak_mem_gb,
        "parameter_count": int(n_params),
    }

    report, per_page_rows, scorecard = _score_northstar(
        pack_dir=pack_dir,
        pred_dir=mxml_dir,
        timing_by_page=latencies,
        telemetry=telemetry,
        limit_pages=LIMIT_PAGES,
    )

    report_path = run_dir / "report.json"
    per_page_path = run_dir / "per_page.json"
    run_ledger_path = run_dir / "run_ledger.json"

    _atomic_write_json(report_path, report)
    _atomic_write_json(
        per_page_path,
        {
            "exp_id": EXP_ID,
            "system_name": SYSTEM_NAME,
            "model_id": MODEL_ID,
            "pages": per_page_rows,
        },
    )
    _atomic_write_json(
        run_ledger_path,
        {
            "exp_id": EXP_ID,
            "system_name": SYSTEM_NAME,
            "model_id": MODEL_ID,
            "gpu": gpu_name,
            "started_at": started_at,
            "completed_at": _utc_now(),
            "elapsed_sec": round(time.monotonic() - t_start, 2),
            "pages_total": len(pages),
            "conversion_ok": n_ok,
            "conversion_fail": n_fail,
            "telemetry": telemetry,
            "headline": report.get("layer1_headline", {}),
            "hf_eval_path": HF_EVAL_PATH,
            "hf_per_page_path": HF_PER_PAGE_PATH,
        },
    )

    print("\n" + scorecard + "\n", flush=True)

    if hf_token and LIMIT_PAGES == 0:
        api = HfApi(token=hf_token)
        for local_p, repo_p, msg in (
            (report_path, HF_EVAL_PATH, f"eval: {EXP_ID} NorthStar 125p report"),
            (per_page_path, HF_PER_PAGE_PATH, f"eval: {EXP_ID} NorthStar 125p per-page"),
            (run_ledger_path, HF_RUN_PATH, f"run: {EXP_ID} ledger"),
        ):
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
