#!/usr/bin/env python3
"""Iter 13 Exp i13-17 on Google Colab L4:
PaddleOCR-VL Valid Region Focus Module (VRFM) Border & Staff-Band Routing
on `suite31_hw` (14 HW + 10 printed pieces × {clean, photo} = 48 pages).

Applies zero-retrain inference-time spatial saliency shaping via a clean forward
hook on `model.router.saliency_net` prior to deterministic top-k selection:
  1. Outer border-ring clutter suppression (`border_frac` in {0.03, 0.06})
  2. Horizontal 3-row smoothed staff-band energy prior (`band_alpha` in {0.30, 0.60})
  3. Combined VRFM + optional `max_keep_ratio=0.55` budget lift.

Expects on Colab VM:
  - /content/bundle.tar.gz (or pre-extracted /content/workspace/sheet-music-converter/tmp-train/mvd)
  - /tmp/.hf_token (Hugging Face token; chmod 600, deleted immediately and in finally)
"""

from __future__ import annotations

import contextlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tarfile
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

EXP_ID = "2026-11-26T07-40-i13-17-paddleocr-vrfm-band-router"
HF_REPO = "gregoreesmaa/sheet-music-converter-omr"
HF_CKPT_CANDIDATES = (
    "checkpoints/iter10_5_best/uni_omr_ema_best.pt",
    "checkpoints/iter10_soup_baseline/uni_omr_soup.pt",
)
HF_SUITE31_TAR = "datasets/suite31_hw_eval.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

TOKEN_PATH = Path("/tmp/.hf_token")
BUNDLE_PATH = Path("/content/bundle.tar.gz")
WORK_ROOT = Path("/content/workspace")
CACHE_ROOT = Path("/content/cache")

ARMS: List[Dict[str, Any]] = [
    {"key": "baseline_global_topk", "border_frac": 0.00, "band_alpha": 0.00, "keep_ratio": None},
    {"key": "vrfm_border_3pct", "border_frac": 0.03, "band_alpha": 0.00, "keep_ratio": None},
    {"key": "vrfm_border_6pct", "border_frac": 0.06, "band_alpha": 0.00, "keep_ratio": None},
    {"key": "vrfm_band_0.30", "border_frac": 0.00, "band_alpha": 0.30, "keep_ratio": None},
    {"key": "vrfm_band_0.60", "border_frac": 0.00, "band_alpha": 0.60, "keep_ratio": None},
    {"key": "vrfm_b3_band0.30", "border_frac": 0.03, "band_alpha": 0.30, "keep_ratio": None},
    {"key": "vrfm_b3_band0.30_k55", "border_frac": 0.03, "band_alpha": 0.30, "keep_ratio": 0.55},
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _read_and_purge_hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token and TOKEN_PATH.exists():
        try:
            os.chmod(TOKEN_PATH, 0o600)
        except OSError:
            pass
        token = TOKEN_PATH.read_text(encoding="utf-8").strip()
    if TOKEN_PATH.exists():
        try:
            TOKEN_PATH.unlink()
        except OSError:
            pass
    if not token:
        raise RuntimeError("HF token not found in /tmp/.hf_token or HF_TOKEN env var")
    return token


def _setup_environment() -> Path:
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    mvd_dir = WORK_ROOT / "sheet-music-converter" / "tmp-train" / "mvd"
    if not mvd_dir.exists():
        alt_mvd = Path("/workspace/sheet-music-converter/tmp-train/mvd")
        if alt_mvd.exists():
            mvd_dir = alt_mvd
        else:
            if not BUNDLE_PATH.exists():
                raise FileNotFoundError(f"Missing bundle at {BUNDLE_PATH}")
            print(f"[{EXP_ID}] Extracting {BUNDLE_PATH} -> {WORK_ROOT}...", flush=True)
            with tarfile.open(BUNDLE_PATH, "r:*") as tf:
                tf.extractall(path=WORK_ROOT)

    needed = []
    for mod, pkg in [
        ("huggingface_hub", "huggingface_hub>=0.26"),
        ("PIL", "pillow"),
        ("numpy", "numpy"),
    ]:
        try:
            __import__(mod)
        except ImportError:
            needed.append(pkg)
    if needed:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *needed], check=True)
    return mvd_dir


def _ensure_assets(hf_token: str) -> Tuple[Path, str, Path]:
    from huggingface_hub import hf_hub_download

    ckpt_path: Optional[Path] = None
    ckpt_hf_used = ""
    last_err: Optional[Exception] = None
    for cand in HF_CKPT_CANDIDATES:
        local_p = CACHE_ROOT / cand
        if local_p.exists():
            ckpt_path = local_p
            ckpt_hf_used = cand
            break
        try:
            print(f"[{EXP_ID}] Downloading {cand}...", flush=True)
            hf_hub_download(
                repo_id=HF_REPO,
                repo_type="dataset",
                filename=cand,
                local_dir=str(CACHE_ROOT),
                token=hf_token,
            )
            ckpt_path = local_p
            ckpt_hf_used = cand
            break
        except Exception as exc:
            last_err = exc
            print(f"[{EXP_ID}] Candidate {cand} unavailable ({exc}); trying fallback...", flush=True)
    if ckpt_path is None:
        raise RuntimeError(f"Could not download any checkpoint: {last_err}")

    s31_dir = CACHE_ROOT / "datasets" / "suite31_hw_eval"
    sentinel = s31_dir / ".extracted"
    if not sentinel.exists():
        print(f"[{EXP_ID}] Downloading {HF_SUITE31_TAR}...", flush=True)
        tar_p = Path(
            hf_hub_download(
                repo_id=HF_REPO,
                repo_type="dataset",
                filename=HF_SUITE31_TAR,
                local_dir=str(CACHE_ROOT / "tars"),
                token=hf_token,
            )
        )
        s31_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_p, "r:*") as tf:
            tf.extractall(path=s31_dir)
        sentinel.write_text(_utc_now() + "\n", encoding="utf-8")
        tar_p.unlink(missing_ok=True)

    manifests = sorted(s31_dir.rglob("suite31_hw_manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"Missing suite31_hw_manifest.json under {s31_dir}")
    return ckpt_path, ckpt_hf_used, manifests[0]


def _extract_musicxml_notes(xml_path: Path) -> List[Tuple[str, int, str, str, str]]:
    notes: List[Tuple[str, int, str, str, str]] = []
    try:
        root = ET.parse(str(xml_path)).getroot()
    except Exception:
        return notes

    def _strip(tag: str) -> str:
        return tag.split("}", 1)[-1] if "}" in tag else tag

    for elem in root.iter():
        if _strip(elem.tag) != "note":
            continue
        is_rest = is_grace = False
        step = octave = alter = dur = ""
        n_dots = 0
        for child in elem:
            ctag = _strip(child.tag)
            if ctag == "rest":
                is_rest = True
            elif ctag == "grace":
                is_grace = True
            elif ctag == "pitch":
                for pc in child:
                    ptag = _strip(pc.tag)
                    text = (pc.text or "").strip()
                    if ptag == "step":
                        step = text.upper()
                    elif ptag == "octave":
                        octave = text
                    elif ptag == "alter":
                        try:
                            av = int(round(float(text)))
                            if av != 0:
                                alter = f"a{av}"
                        except ValueError:
                            pass
            elif ctag == "type":
                dur = (child.text or "").strip().lower()
            elif ctag == "dot":
                n_dots += 1
        if not is_rest and not is_grace and step and octave and dur:
            notes.append((dur, n_dots, step, octave, alter))
    return notes


@contextlib.contextmanager
def _vrfm_saliency_context(
    model: Any,
    *,
    border_frac: float,
    band_alpha: float,
    keep_ratio: Optional[float],
    eval_s31: Any,
) -> Iterator[None]:
    import torch.nn.functional as F

    router = eval_s31._get_model_router(model)
    saliency_net = getattr(router, "saliency_net", None) if router is not None else None
    handle = None

    if saliency_net is not None and (border_frac > 0.0 or band_alpha > 0.0):
        def _hook(_mod: Any, _inp: Any, out: Any) -> Any:
            y = out.clone()
            _, _, h_p, w_p = y.shape
            if band_alpha > 0.0 and h_p >= 3:
                row_mean = y.mean(dim=3, keepdim=True)
                row_smooth = F.avg_pool2d(
                    row_mean, kernel_size=(3, 1), stride=1, padding=(1, 0), count_include_pad=False
                )
                y = y + band_alpha * row_smooth
            if border_frac > 0.0:
                bh = max(1, int(round(h_p * border_frac)))
                bw = max(1, int(round(w_p * border_frac)))
                if bh * 2 < h_p:
                    y[:, :, :bh, :] = y[:, :, :bh, :] - 6.0
                    y[:, :, -bh:, :] = y[:, :, -bh:, :] - 6.0
                if bw * 2 < w_p:
                    y[:, :, :, :bw] = y[:, :, :, :bw] - 6.0
                    y[:, :, :, -bw:] = y[:, :, :, -bw:] - 6.0
            return y

        handle = saliency_net.register_forward_hook(_hook)

    try:
        with eval_s31.override_max_keep_ratio(model, keep_ratio):
            yield
    finally:
        if handle is not None:
            handle.remove()


def _decode_arm(
    model: Any,
    cfg: Any,
    img_t: Any,
    ref_notes: Sequence[Tuple[str, int, str, str, str]],
    arm: Dict[str, Any],
    *,
    max_len: int,
    use_bf16: bool,
    device: Any,
    eval_s31: Any,
    rlvr_grpo: Any,
) -> Dict[str, Any]:
    import torch

    _, _, eos_id, _, _, id_to_tok = eval_s31._vocab_spec(int(cfg.vocab_size))
    t0 = time.perf_counter()
    with _vrfm_saliency_context(
        model,
        border_frac=float(arm["border_frac"]),
        band_alpha=float(arm["band_alpha"]),
        keep_ratio=arm["keep_ratio"],
        eval_s31=eval_s31,
    ):
        with torch.no_grad():
            if use_bf16 and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    seq, mtp_stats = eval_s31.generate_mtp_spec(
                        model, img_t, max_len=max_len, use_grammar=True,
                        strict_grammar=False, return_stats=True,
                    )
            else:
                seq, mtp_stats = eval_s31.generate_mtp_spec(
                    model, img_t, max_len=max_len, use_grammar=True,
                    strict_grammar=False, return_stats=True,
                )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = max(1e-6, time.perf_counter() - t0)

    seq_ids = seq[0].detach().cpu().tolist()
    has_eos = eos_id in seq_ids[1:]
    pred_raw = eval_s31.decode_uni_compat(
        seq_ids, vocab_size=int(cfg.vocab_size), skip_special=True
    )
    if not has_eos:
        pred_raw = eval_s31.trim_truncated_tokens(pred_raw)
    hyp_toks, n_sys = eval_s31.stitch_uni_tokens(pred_raw)
    struct = rlvr_grpo.compute_unlabeled_structural_reward(
        pred_raw, vocab=id_to_tok, terminated_eos=has_eos
    )
    hyp_notes = rlvr_grpo.extract_pitch_dur_notes(hyp_toks)
    note_f1 = rlvr_grpo.compute_sequence_f1(ref_notes, hyp_notes)
    return {
        "note_f1_py": round(float(note_f1), 6),
        "struct_reward": round(float(struct["reward"]), 6),
        "r_rhythm": round(float(struct["r_rhythm"]), 4),
        "has_eos": bool(has_eos),
        "n_sys": int(n_sys),
        "n_hyp_notes": len(hyp_notes),
        "elapsed_secs": round(elapsed, 3),
        "mtp_acc": round(float(mtp_stats.get("acceptance_rate", 0.0)), 4),
    }


def _summarize_rows(rows: Sequence[Dict[str, Any]], arm_keys: Sequence[str]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for domain in ("clean", "photo"):
        drows = [r for r in rows if r["domain"] == domain]
        base_vals = [float(r["arms"]["baseline_global_topk"]["note_f1_py"]) for r in drows if "baseline_global_topk" in r.get("arms", {})]
        base_mean = round(sum(base_vals) / len(base_vals), 6) if base_vals else None
        arms_out: Dict[str, Any] = {}
        for akey in arm_keys:
            vals = [float(r["arms"][akey]["note_f1_py"]) for r in drows if akey in r.get("arms", {})]
            times = [float(r["arms"][akey]["elapsed_secs"]) for r in drows if akey in r.get("arms", {})]
            wins = sum(
                1 for r in drows
                if akey in r.get("arms", {}) and "baseline_global_topk" in r.get("arms", {})
                and float(r["arms"][akey]["note_f1_py"]) > float(r["arms"]["baseline_global_topk"]["note_f1_py"]) + 1e-6
            )
            losses = sum(
                1 for r in drows
                if akey in r.get("arms", {}) and "baseline_global_topk" in r.get("arms", {})
                and float(r["arms"][akey]["note_f1_py"]) < float(r["arms"]["baseline_global_topk"]["note_f1_py"]) - 1e-6
            )
            mean_v = round(sum(vals) / len(vals), 6) if vals else None
            arms_out[akey] = {
                "n": len(vals),
                "note_f1_py": mean_v,
                "delta_note_f1_py": round(mean_v - base_mean, 6) if (mean_v is not None and base_mean is not None) else None,
                "wins": wins,
                "ties": len(vals) - wins - losses,
                "losses": losses,
                "median_secs": round(float(statistics.median(times)), 3) if times else None,
            }
        summary[domain] = {"n": len(drows), "baseline_note_f1_py": base_mean, "arms": arms_out}
    return summary


def run_sweep(
    max_dim: int = 1280,
    max_len: int = 3840,
    printed_n: int = 10,
    save_every_minutes: float = 5.0,
) -> Dict[str, Any]:
    try:
        hf_token = _read_and_purge_hf_token()
        mvd_dir = _setup_environment()
        if str(mvd_dir) not in sys.path:
            sys.path.insert(0, str(mvd_dir))

        from huggingface_hub import HfApi
        from PIL import Image
        import torch
        import eval_uni_suite31 as eval_s31
        import rlvr_grpo
        import uni_omr

        t_start = time.perf_counter()
        started_utc = _utc_now()
        if torch.cuda.is_available():
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
        cc = torch.cuda.get_device_capability(0) if device.type == "cuda" else (0, 0)
        use_bf16 = bool(device.type == "cuda" and cc[0] >= 8)
        platform_str = f"Colab {gpu_name} (cc={cc[0]}.{cc[1]}, bf16={use_bf16})"
        print(f"[{EXP_ID}] Running on {platform_str}", flush=True)

        ckpt_path, ckpt_hf_used, manifest_path = _ensure_assets(hf_token)
        api = HfApi(token=hf_token)

        ledger_tmp = CACHE_ROOT / "runs" / f"{EXP_ID}.json"
        _atomic_write_json(
            ledger_tmp,
            {"exp_id": EXP_ID, "status": "RUNNING", "started_utc": started_utc, "platform": platform_str},
        )
        api.upload_file(
            path_or_fileobj=str(ledger_tmp),
            path_in_repo=HF_RUN_PATH,
            repo_id=HF_REPO,
            repo_type="dataset",
            commit_message=f"{EXP_ID}: mark RUNNING",
        )

        model, cfg, ckpt_meta = eval_s31.load_uni_checkpoint(str(ckpt_path), device=device)
        model.eval()

        _, all_pieces = eval_s31.resolve_manifest_pieces(manifest_path)
        hw_pieces = [p for p in all_pieces if p["handwritten"]]
        pr_pieces = [p for p in all_pieces if not p["handwritten"]][:printed_n]
        arm_keys = [a["key"] for a in ARMS]

        partial_path = CACHE_ROOT / "evals" / f"{EXP_ID}_partial.json"
        hw_rows: List[Dict[str, Any]] = []
        pr_rows: List[Dict[str, Any]] = []
        last_save_t = time.perf_counter()

        def _eval_split(pieces: List[Dict[str, Any]], tag: str, out_rows: List[Dict[str, Any]]) -> None:
            nonlocal last_save_t
            for idx, piece in enumerate(pieces, 1):
                slug = piece["slug"]
                ref_notes = _extract_musicxml_notes(piece["gt"])
                for domain in ("clean", "photo"):
                    with Image.open(piece[domain]) as raw_im:
                        pil_im = uni_omr._resize_fullpage_pil(raw_im.convert("RGB"), max_dim)
                    img_t = uni_omr._pil_to_chw_tensor(pil_im).unsqueeze(0).to(device)

                    arm_results: Dict[str, Any] = {}
                    for arm in ARMS:
                        arm_results[arm["key"]] = _decode_arm(
                            model, cfg, img_t, ref_notes, arm,
                            max_len=max_len, use_bf16=use_bf16, device=device,
                            eval_s31=eval_s31, rlvr_grpo=rlvr_grpo,
                        )

                    out_rows.append({"piece": slug, "domain": domain, "arms": arm_results})
                    base_f1 = arm_results["baseline_global_topk"]["note_f1_py"]
                    deltas = " ".join(
                        f"{k}:{arm_results[k]['note_f1_py'] - base_f1:+.4f}"
                        for k in arm_keys if k != "baseline_global_topk"
                    )
                    print(f"[{tag} {idx:02d}/{len(pieces):02d}] {slug:34s} [{domain:5s}] base={base_f1:.4f} | {deltas}", flush=True)

                    if (time.perf_counter() - last_save_t) >= save_every_minutes * 60.0 or domain == "photo":
                        _atomic_write_json(
                            partial_path,
                            {
                                "exp_id": EXP_ID,
                                "status": "RUNNING",
                                "hw_summary": _summarize_rows(hw_rows, arm_keys),
                                "printed_summary": _summarize_rows(pr_rows, arm_keys),
                                "hw_rows": hw_rows,
                                "printed_rows": pr_rows,
                            },
                        )
                        last_save_t = time.perf_counter()

        _eval_split(hw_pieces, "HW", hw_rows)
        _eval_split(pr_pieces, "PR", pr_rows)

        hw_summary = _summarize_rows(hw_rows, arm_keys)
        pr_summary = _summarize_rows(pr_rows, arm_keys)
        non_base = [k for k in arm_keys if k != "baseline_global_topk"]
        best_arm = max(
            non_base,
            key=lambda k: 0.5 * (
                (hw_summary["clean"]["arms"][k]["delta_note_f1_py"] or 0.0)
                + (hw_summary["photo"]["arms"][k]["delta_note_f1_py"] or 0.0)
            ),
        )
        total_secs = round(time.perf_counter() - t_start, 2)
        final_report = {
            "exp_id": EXP_ID,
            "status": "COMPLETED",
            "started_utc": started_utc,
            "completed_utc": _utc_now(),
            "elapsed_secs": total_secs,
            "platform": platform_str,
            "base_ckpt": ckpt_hf_used,
            "base_ckpt_meta": ckpt_meta,
            "config": {"arms": ARMS, "max_dim": max_dim, "max_len": max_len},
            "best_arm_key": best_arm,
            "hw_summary": hw_summary,
            "printed_summary": pr_summary,
            "hw_rows": hw_rows,
            "printed_rows": pr_rows,
        }
        final_local = CACHE_ROOT / "evals" / f"{EXP_ID}.json"
        _atomic_write_json(final_local, final_report)

        api.upload_file(
            path_or_fileobj=str(final_local),
            path_in_repo=HF_EVAL_PATH,
            repo_id=HF_REPO,
            repo_type="dataset",
            commit_message=f"{EXP_ID}: upload PaddleOCR-VL VRFM band router eval JSON",
        )
        _atomic_write_json(
            ledger_tmp,
            {
                "exp_id": EXP_ID,
                "status": "COMPLETED",
                "started_utc": started_utc,
                "completed_utc": final_report["completed_utc"],
                "elapsed_secs": total_secs,
                "platform": platform_str,
                "best_arm_key": best_arm,
                "hf_eval_path": HF_EVAL_PATH,
            },
        )
        api.upload_file(
            path_or_fileobj=str(ledger_tmp),
            path_in_repo=HF_RUN_PATH,
            repo_id=HF_REPO,
            repo_type="dataset",
            commit_message=f"{EXP_ID}: mark COMPLETED",
        )
        print(f"[{EXP_ID}] COMPLETED in {total_secs}s | best_arm={best_arm}", flush=True)
        return final_report
    finally:
        if TOKEN_PATH.exists():
            try:
                TOKEN_PATH.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    run_sweep()
