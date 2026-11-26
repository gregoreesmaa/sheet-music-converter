#!/usr/bin/env python3
"""Modal L4 / Colab L4 evaluator for Qwen2.5-VL-7B-Instruct (VLM -> ABC -> MusicXML) on OMR-NorthStar (125 pages).

Pipeline:
  1. Download & unpack self-contained `datasets/northstar_eval_pack_125p.tar.gz` from HF Private Hub
     (125 pages: 31 CDR + 31 ALS + 31 RWP + 28 HWM + 4 Pillar 3B Negative Controls + `omr_northstar`).
  2. Load `Qwen/Qwen2.5-VL-7B-Instruct` (Apache-2.0, ungated) in BF16 on CUDA L4 (24 GB).
  3. Run deterministic greedy transcription -> ABC notation (or `NO_MUSIC` abstention) -> `music21` -> `.musicxml`.
  4. Score all 125 pages with the official `omr_northstar` 4-layer scorer.
  5. Upload `evals/2026-11-26T15-45-eval-northstar-qwen25vl.json` (plus `_per_page.json` and `runs/*.json`)
     to `gregoreesmaa/sheet-music-converter-omr`.

Note: `guangyangmusic/legato` / `legato-small` / `legato-2` are gated (403 GatedRepoError) on HF;
this script evaluates the ungated `Qwen/Qwen2.5-VL-7B-Instruct` baseline.

Never echoes secrets (`HF_TOKEN` is read from env or `/tmp/.hf_token`).
"""

from __future__ import annotations

import gc
import json
import os
import pathlib
import re
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

EXP_ID = "2026-11-26T15-45-eval-northstar-qwen25vl"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_PACK_TAR = "datasets/northstar_eval_pack_125p.tar.gz"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_PER_PAGE_PATH = f"evals/{EXP_ID}_per_page.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

MODEL_ID = os.environ.get("QWEN_VL_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
SYSTEM_NAME = os.environ.get("NORTHSTAR_SYSTEM_NAME", "qwen25vl-7b-abc")
MAX_NEW_TOKENS = int(os.environ.get("QWEN_VL_MAX_NEW_TOKENS", "2048"))
LIMIT_PAGES = int(os.environ.get("NORTHSTAR_LIMIT", "0"))  # 0 = all 125 pages

OMR_ABC_PROMPT = (
    "You are an expert Optical Music Recognition (OMR) system.\n"
    "Examine the image carefully:\n"
    "- If the image does NOT contain any sheet music or musical notation (for example, a blank page, "
    "a text-only page, or a non-music photograph), output ONLY the exact token: NO_MUSIC\n"
    "- Otherwise, transcribe the ENTIRE sheet music page accurately into valid ABC music notation.\n"
    "Include required headers (X:1, M:<meter>, L:<unit note length>, K:<key>) and all staves/voices "
    "(using V:1, V:2, etc. if polyphonic/grand-staff), pitches, accidentals, durations, rests, and barlines.\n"
    "Output ONLY raw ABC notation with no explanation or commentary."
)

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
    """Ensure transformers>=4.49.0 (Qwen2.5-VL support), accelerate, qwen-vl-utils, and music21 are installed."""
    need_install: List[str] = []
    try:
        import transformers

        ver_parts = tuple(int(x) for x in transformers.__version__.split(".")[:2] if x.isdigit())
        if ver_parts < (4, 49):
            need_install.append("transformers>=4.49.0")
    except ImportError:
        need_install.append("transformers>=4.49.0")

    for mod, pkg in (
        ("accelerate", "accelerate"),
        ("qwen_vl_utils", "qwen-vl-utils"),
        ("music21", "music21"),
        ("huggingface_hub", "huggingface_hub>=0.26"),
    ):
        try:
            __import__(mod)
        except ImportError:
            need_install.append(pkg)

    if need_install:
        print(f"[setup] Installing packages: {need_install}", flush=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--upgrade-strategy", "only-if-needed", *need_install],
            check=True,
        )


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


def _load_qwen_vl(model_id: str):
    """Load Qwen2.5-VL-7B-Instruct in native BF16 with PyTorch SDPA."""
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else (torch.float16 if torch.cuda.is_available() else torch.float32)

    processor = AutoProcessor.from_pretrained(
        model_id,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[model] Loaded {model_id} | dtype={dtype} | params={n_params/1e9:.2f}B | attn=sdpa",
        flush=True,
    )
    return model, processor, n_params


def _extract_abc(raw_text: str) -> Tuple[bool, str]:
    """Return `(is_no_music, cleaned_abc_text)` from VLM response."""
    stripped = raw_text.strip()
    if not stripped:
        return False, ""

    # Detect explicit NO_MUSIC abstention (for Pillar 3B negative controls).
    first_line = stripped.splitlines()[0].strip().strip("`").strip()
    if first_line.upper() == "NO_MUSIC" or stripped.upper() == "NO_MUSIC":
        return True, "NO_MUSIC"

    # Strip markdown code fences if the model wrapped ABC in ```abc ... ```
    fence = re.search(r"```(?:abc)?\s*\n(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    abc = fence.group(1).strip() if fence else stripped

    if abc.upper() == "NO_MUSIC":
        return True, "NO_MUSIC"

    # Trim any conversational preamble before `X:` header if present.
    m_hdr = re.search(r"^X:\s*\d+", abc, flags=re.MULTILINE)
    if m_hdr:
        abc = abc[m_hdr.start() :].strip()

    # Ensure minimal ABC header fields exist so music21 can attempt parsing.
    if not re.search(r"^X:", abc, flags=re.MULTILINE):
        abc = "X:1\n" + abc
    if not re.search(r"^K:", abc, flags=re.MULTILINE):
        lines = abc.splitlines()
        insert_at = 1
        while insert_at < len(lines) and re.match(r"^[A-Za-z]:", lines[insert_at]):
            insert_at += 1
        lines.insert(insert_at, "K:C")
        abc = "\n".join(lines)

    return False, abc + "\n"


def _abc_to_musicxml(is_no_music: bool, abc_text: str, out_mxml_path: pathlib.Path) -> Tuple[bool, str]:
    """Convert ABC notation to MusicXML via music21 (or emit valid empty MusicXML on NO_MUSIC)."""
    if is_no_music:
        _atomic_write_text(out_mxml_path, EMPTY_VALID_MUSICXML)
        return True, "no-music-abstain"

    if not abc_text.strip():
        _atomic_write_text(out_mxml_path, "")
        return False, "empty-abc"

    try:
        import music21

        score = music21.converter.parseData(abc_text, format="abc")
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

    import torch
    from PIL import Image
    from huggingface_hub import HfApi

    hf_token = _get_hf_token()
    vol_root = _get_vol_root()
    pack_dir = _ensure_northstar_pack(vol_root, hf_token)

    run_dir = vol_root / "runs" / EXP_ID
    mxml_dir = run_dir / "musicxml"
    abc_dir = run_dir / "abc"
    mxml_dir.mkdir(parents=True, exist_ok=True)
    abc_dir.mkdir(parents=True, exist_ok=True)
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

    model, processor, n_params = _load_qwen_vl(MODEL_ID)

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
        out_abc = abc_dir / f"{pid}.abc"

        if out_mxml.is_file() and pid in latencies:
            status = conv_statuses.get(pid, "resumed")
            if status in ("ok", "no-music-abstain"):
                n_ok += 1
            else:
                n_fail += 1
            continue

        img_path = pack_dir / str(rec["image_path"])
        t0 = time.perf_counter()
        try:
            pil_img = Image.open(img_path).convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_img},
                        {"type": "text", "text": OMR_ABC_PROMPT},
                    ],
                }
            ]
            text_prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text_prompt],
                images=[pil_img],
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                )
            trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            raw_out = processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            is_no_music, abc_text = _extract_abc(raw_out)
            _atomic_write_text(out_abc, abc_text if abc_text else raw_out)
            ok, status = _abc_to_musicxml(is_no_music, abc_text, out_mxml)
        except Exception as exc:
            _atomic_write_text(out_abc, f"% ERROR: {type(exc).__name__}: {exc}\n")
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
    del processor
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
