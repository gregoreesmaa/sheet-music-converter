#!/usr/bin/env python3
"""Iter 13 (i13-10) on Google Colab L4:
OEMER/muskew Staffline-Aware Vertical Dewarping & Homr `unit_size` Scale Normalization
ablation on full 45-piece `suite31_hw` (90 pages: 31 printed + 14 HW, clean + photo).

Expects:
  - /content/bundle.tar.gz (or pre-extracted sheet-music-converter/tmp-train/mvd)
  - /tmp/.hf_token (Hugging Face token; chmod 600, deleted immediately and in finally)
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tarfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

EXP_ID = "2026-11-26T07-40-i13-10-oemer-staff-dewarp-unitsize"
HF_REPO = "gregoreesmaa/sheet-music-converter-omr"
HF_CKPT_PATH = os.environ.get(
    "HF_CKPT_PATH",
    "checkpoints/iter10_soup_baseline/uni_omr_soup.pt",
)
HF_SUITE31_TAR = "datasets/suite31_hw_eval.tar"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

TOKEN_PATH = Path("/tmp/.hf_token")
BUNDLE_CANDIDATES = (
    Path("/content/bundle.tar.gz"),
    Path("/content/ov_mvd_bundle.tar.gz"),
)
WORK_ROOT = Path("/content/workspace")
CACHE_ROOT = Path("/content/cache")

ARMS: List[Dict[str, Any]] = [
    {"key": "baseline_1280", "dewarp": False, "unitsize_norm": False},
    {"key": "oemer_dewarp_only", "dewarp": True, "unitsize_norm": False},
    {"key": "homr_unitsize_only", "dewarp": False, "unitsize_norm": True},
    {"key": "oemer_dewarp_plus_unitsize", "dewarp": True, "unitsize_norm": True},
]

BASE_MAX_DIM = 1280
MIN_UNITSIZE_DIM = 960
MAX_UNITSIZE_DIM = 1600
TARGET_UNIT_SIZE_PX = 7.5
MAX_LEN = 3840
SAVE_EVERY_MINUTES = 5.0


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

    candidates = (
        WORK_ROOT / "sheet-music-converter" / "tmp-train" / "mvd",
        Path("/workspace/sheet-music-converter/tmp-train/mvd"),
        Path("/content/sheet-music-converter/tmp-train/mvd"),
        Path.cwd() / "tmp-train" / "mvd",
    )
    mvd_dir: Optional[Path] = None
    for cand in candidates:
        if (cand / "eval_uni_suite31.py").is_file():
            mvd_dir = cand
            break

    if mvd_dir is None:
        for bundle in BUNDLE_CANDIDATES:
            if bundle.is_file():
                print(f"[{EXP_ID}] Extracting {bundle} -> {WORK_ROOT}...", flush=True)
                with tarfile.open(bundle, "r:*") as tf:
                    tf.extractall(path=WORK_ROOT)
                for cand in (
                    WORK_ROOT / "sheet-music-converter" / "tmp-train" / "mvd",
                    WORK_ROOT / "tmp-train" / "mvd",
                    WORK_ROOT / "mvd",
                ):
                    if (cand / "eval_uni_suite31.py").is_file():
                        mvd_dir = cand
                        break
                if mvd_dir is not None:
                    break

    if mvd_dir is None:
        raise FileNotFoundError("Could not locate tmp-train/mvd/eval_uni_suite31.py")

    needed: List[str] = []
    for mod, pkg in (
        ("huggingface_hub", "huggingface_hub>=0.26"),
        ("PIL", "pillow"),
        ("numpy", "numpy"),
        ("cv2", "opencv-python-headless"),
        ("scipy", "scipy"),
        ("tqdm", "tqdm"),
    ):
        try:
            __import__(mod)
        except ImportError:
            needed.append(pkg)
    if needed:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *needed], check=True)
    return mvd_dir


def _ensure_assets(hf_token: str) -> Tuple[Path, Path]:
    from huggingface_hub import hf_hub_download

    ckpt_path = CACHE_ROOT / HF_CKPT_PATH
    if not ckpt_path.exists():
        print(f"[{EXP_ID}] Downloading {HF_CKPT_PATH}...", flush=True)
        hf_hub_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            filename=HF_CKPT_PATH,
            local_dir=str(CACHE_ROOT),
            token=hf_token,
        )

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
    return ckpt_path, manifests[0]


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


# ---------------------------------------------------------------------------
# Competitor-Inspired Front-End: OEMER Vertical Dewarp + Homr unit_size Norm
# ---------------------------------------------------------------------------

def oemer_staffline_vertical_dewarp(img_pil: Any) -> Tuple[Any, Dict[str, float]]:
    """OEMER/muskew-inspired staffline-aware vertical y(x) dewarping.

    Extracts horizontal staff-line ridge response via directional Sobel/morphology,
    tracks vertical phase shifts across horizontal stripes relative to the center
    column via 1D cross-correlation, fits a smooth clamped displacement curve y(x)
    (|dy| <= 3% H), and remaps columns with replicate paper-border fill.
    """
    import cv2
    import numpy as np
    from PIL import Image
    from scipy.ndimage import gaussian_filter1d

    t0 = time.perf_counter()
    gray = np.asarray(img_pil.convert("L"), dtype=np.float32)
    h, w = gray.shape
    if h < 128 or w < 128:
        return img_pil, {"dewarp_ms": 0.0, "max_dy_px": 0.0}

    # Invert so dark ink is positive; isolate horizontal staffline ridges
    inv = 255.0 - gray
    k_len = max(15, int(round(w * 0.025)))
    horiz_k = cv2.getStructuringElement(cv2.MORPH_RECT, (k_len, 1))
    staff_ridge = cv2.morphologyEx(inv, cv2.MORPH_OPEN, horiz_k)

    n_stripes = 24
    xs = np.linspace(0, w, n_stripes + 1, dtype=np.int32)
    centers = 0.5 * (xs[:-1] + xs[1:])
    profiles = np.zeros((n_stripes, h), dtype=np.float32)
    for i in range(n_stripes):
        strip = staff_ridge[:, xs[i]:xs[i + 1]]
        prof = strip.mean(axis=1)
        prof = prof - prof.mean()
        norm = float(np.linalg.norm(prof))
        if norm > 1e-4:
            profiles[i] = prof / norm

    # Track sequential vertical shifts outward from the center stripe
    max_step = max(4, int(round(h * 0.008)))
    shifts = np.zeros(n_stripes, dtype=np.float32)
    mid = n_stripes // 2

    def _best_shift(ref: np.ndarray, cur: np.ndarray) -> float:
        if np.linalg.norm(ref) < 1e-4 or np.linalg.norm(cur) < 1e-4:
            return 0.0
        best_s = 0
        best_v = -1.0
        for s in range(-max_step, max_step + 1):
            if s >= 0:
                val = float(np.dot(ref[: h - s], cur[s:]))
            else:
                val = float(np.dot(ref[-s:], cur[: h + s]))
            if val > best_v:
                best_v = val
                best_s = s
        return float(best_s) if best_v > 0.15 else 0.0

    for i in range(mid + 1, n_stripes):
        shifts[i] = shifts[i - 1] + _best_shift(profiles[i - 1], profiles[i])
    for i in range(mid - 1, -1, -1):
        shifts[i] = shifts[i + 1] + _best_shift(profiles[i + 1], profiles[i])

    # Remove median offset, smooth, and clamp to <= 3% page height
    shifts = shifts - float(np.median(shifts))
    shifts = gaussian_filter1d(shifts, sigma=1.8, mode="nearest")
    max_dy = float(h * 0.03)
    shifts = np.clip(shifts, -max_dy, max_dy)

    col_x = np.arange(w, dtype=np.float32)
    col_dy = np.interp(col_x, centers, shifts).astype(np.float32)

    peak_to_peak = float(col_dy.max() - col_dy.min())
    if peak_to_peak < 1.5:
        ms = (time.perf_counter() - t0) * 1000.0
        return img_pil, {"dewarp_ms": round(ms, 2), "max_dy_px": round(peak_to_peak, 2)}

    map_x = np.broadcast_to(col_x[None, :], (h, w)).astype(np.float32)
    row_y = np.arange(h, dtype=np.float32)[:, None]
    map_y = (row_y + col_dy[None, :]).astype(np.float32)

    rgb = np.asarray(img_pil.convert("RGB"))
    remapped = cv2.remap(
        rgb,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    ms = (time.perf_counter() - t0) * 1000.0
    return Image.fromarray(remapped), {
        "dewarp_ms": round(ms, 2),
        "max_dy_px": round(peak_to_peak, 2),
    }


def homr_estimate_unitsize_max_dim(img_pil: Any) -> Tuple[int, Dict[str, float]]:
    """Homr-inspired `unit_size` (inter-staffline spacing) estimator.

    Measures dominant vertical staffline period at the 1280px reference scale
    using 1D autocorrelation of horizontal ink ridges, then computes an adaptive
    `max_dim` clamped to [MIN_UNITSIZE_DIM, MAX_UNITSIZE_DIM] so dense multi-staff
    pages receive higher resolution while sparse pages avoid overscaling.
    """
    import cv2
    import numpy as np

    t0 = time.perf_counter()
    w0, h0 = img_pil.size
    ref_scale = float(BASE_MAX_DIM) / float(max(w0, h0))
    rw = max(64, int(round(w0 * ref_scale)))
    rh = max(64, int(round(h0 * ref_scale)))
    gray = np.asarray(img_pil.convert("L").resize((rw, rh)), dtype=np.float32)
    inv = 255.0 - gray

    k_len = max(15, int(round(rw * 0.025)))
    horiz_k = cv2.getStructuringElement(cv2.MORPH_RECT, (k_len, 1))
    ridge = cv2.morphologyEx(inv, cv2.MORPH_OPEN, horiz_k)
    prof = ridge.mean(axis=1)
    prof = prof - prof.mean()
    if float(np.linalg.norm(prof)) < 1e-4:
        ms = (time.perf_counter() - t0) * 1000.0
        return BASE_MAX_DIM, {"unitsize_ms": round(ms, 2), "est_unit_px": TARGET_UNIT_SIZE_PX, "target_max_dim": BASE_MAX_DIM}

    ac = np.correlate(prof, prof, mode="full")[len(prof) - 1 :]
    if ac[0] > 1e-6:
        ac = ac / ac[0]

    lo, hi = 4, min(26, len(ac) - 1)
    if hi <= lo:
        ms = (time.perf_counter() - t0) * 1000.0
        return BASE_MAX_DIM, {"unitsize_ms": round(ms, 2), "est_unit_px": TARGET_UNIT_SIZE_PX, "target_max_dim": BASE_MAX_DIM}

    window = ac[lo : hi + 1]
    best_lag = int(lo + int(np.argmax(window)))
    best_corr = float(window[best_lag - lo])
    if best_corr < 0.08:
        est_u = TARGET_UNIT_SIZE_PX
    else:
        est_u = float(best_lag)

    ratio = TARGET_UNIT_SIZE_PX / max(3.5, est_u)
    target_dim = int(round(BASE_MAX_DIM * ratio))
    target_dim = max(MIN_UNITSIZE_DIM, min(MAX_UNITSIZE_DIM, target_dim))
    # Snap to multiple of 32 for clean FPN/axial stride alignment
    target_dim = int(round(target_dim / 32.0) * 32)
    ms = (time.perf_counter() - t0) * 1000.0
    return target_dim, {
        "unitsize_ms": round(ms, 2),
        "est_unit_px": round(est_u, 2),
        "ac_peak": round(best_corr, 4),
        "target_max_dim": target_dim,
    }


def _decode_and_score(
    model: Any,
    cfg: Any,
    img_pil: Any,
    ref_notes: Sequence[Tuple[str, int, str, str, str]],
    *,
    arm: Dict[str, Any],
    use_bf16: bool,
    device: Any,
    eval_s31: Any,
    rlvr_grpo: Any,
) -> Dict[str, Any]:
    import torch

    pre_meta: Dict[str, Any] = {}
    work_img = img_pil
    if arm["dewarp"]:
        work_img, dw_meta = oemer_staffline_vertical_dewarp(work_img)
        pre_meta.update(dw_meta)
    if arm["unitsize_norm"]:
        target_dim, us_meta = homr_estimate_unitsize_max_dim(work_img)
        pre_meta.update(us_meta)
    else:
        target_dim = BASE_MAX_DIM
        pre_meta["target_max_dim"] = target_dim

    img_t = eval_s31.load_page_image_tensor(work_img, max_dim=target_dim).to(device)
    _, _, eos_id, _, _, id_to_tok = eval_s31._vocab_spec(int(cfg.vocab_size))
    model.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        if use_bf16 and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                seq, mtp_stats = eval_s31.generate_mtp_spec(
                    model,
                    img_t,
                    max_len=MAX_LEN,
                    use_grammar=True,
                    strict_grammar=False,
                    return_stats=True,
                )
        else:
            seq, mtp_stats = eval_s31.generate_mtp_spec(
                model,
                img_t,
                max_len=MAX_LEN,
                use_grammar=True,
                strict_grammar=False,
                return_stats=True,
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
        "r_rhythm": round(float(struct["r_rhythm"]), 4),
        "has_eos": bool(has_eos),
        "n_sys": int(n_sys),
        "n_hyp_notes": len(hyp_notes),
        "decode_secs": round(elapsed, 3),
        "mtp_acc": round(float(mtp_stats["acceptance_rate"]), 4),
        "pre_meta": pre_meta,
    }


def _summarize(rows: Sequence[Dict[str, Any]], arm_keys: Sequence[str]) -> Dict[str, Any]:
    slices = {
        "all45_clean": lambda r: r["domain"] == "clean",
        "all45_photo": lambda r: r["domain"] == "photo",
        "pr_clean": lambda r: (not r["handwritten"]) and r["domain"] == "clean",
        "pr_photo": lambda r: (not r["handwritten"]) and r["domain"] == "photo",
        "hw_clean": lambda r: r["handwritten"] and r["domain"] == "clean",
        "hw_photo": lambda r: r["handwritten"] and r["domain"] == "photo",
    }
    out: Dict[str, Any] = {}
    for sname, pred in slices.items():
        srows = [r for r in rows if pred(r)]
        arm_stats: Dict[str, Any] = {}
        base_vals = [float(r["arms"]["baseline_1280"]["note_f1_py"]) for r in srows if "baseline_1280" in r["arms"]]
        base_mean = round(sum(base_vals) / len(base_vals), 6) if base_vals else None
        for k in arm_keys:
            vals = [float(r["arms"][k]["note_f1_py"]) for r in srows if k in r["arms"]]
            rhys = [float(r["arms"][k]["r_rhythm"]) for r in srows if k in r["arms"]]
            secs = [float(r["arms"][k]["decode_secs"]) for r in srows if k in r["arms"]]
            m = round(sum(vals) / len(vals), 6) if vals else None
            arm_stats[k] = {
                "n": len(vals),
                "note_f1_py": m,
                "delta_vs_baseline": round(m - base_mean, 6) if (m is not None and base_mean is not None) else None,
                "r_rhythm_mean": round(sum(rhys) / len(rhys), 4) if rhys else None,
                "median_decode_secs": round(float(statistics.median(secs)), 3) if secs else None,
            }
        out[sname] = {"n": len(srows), "arms": arm_stats}
    return out


def main() -> int:
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

        t_start = time.perf_counter()
        started_utc = _utc_now()
        if torch.cuda.is_available():
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_flash_sdp(True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
        cc = torch.cuda.get_device_capability(0) if device.type == "cuda" else (0, 0)
        use_bf16 = bool(device.type == "cuda" and cc[0] >= 8)
        platform_str = f"colab-{gpu_name}"
        print(f"[{EXP_ID}] {platform_str} (cc={cc[0]}.{cc[1]}, bf16={use_bf16})", flush=True)

        api = HfApi(token=hf_token)
        ckpt_path, manifest_path = _ensure_assets(hf_token)
        model, cfg, ckpt_meta = eval_s31.load_uni_checkpoint(ckpt_path, device=device)
        _, all_pieces = eval_s31.resolve_manifest_pieces(manifest_path)

        partial_path = CACHE_ROOT / "partials" / f"{EXP_ID}_partial.json"
        rows: List[Dict[str, Any]] = []
        done_keys: set[Tuple[str, str]] = set()
        if partial_path.is_file():
            prev = json.loads(partial_path.read_text(encoding="utf-8"))
            rows = list(prev.get("rows", []))
            for r in rows:
                done_keys.add((str(r["piece"]), str(r["domain"])))
            print(f"[{EXP_ID}] Resumed {len(done_keys)}/90 pages from {partial_path}", flush=True)

        arm_keys = [a["key"] for a in ARMS]
        last_save = time.perf_counter()

        for idx, piece in enumerate(all_pieces, start=1):
            slug = str(piece["slug"])
            is_hw = bool(piece["handwritten"])
            ref_notes = _extract_musicxml_notes(Path(piece["gt_xml"]))
            for domain in ("clean", "photo"):
                if (slug, domain) in done_keys:
                    continue
                img_path = Path(piece["clean_img"] if domain == "clean" else piece["photo_img"])
                img_pil = Image.open(img_path).convert("RGB")
                arm_res: Dict[str, Any] = {}
                for arm in ARMS:
                    arm_res[arm["key"]] = _decode_and_score(
                        model,
                        cfg,
                        img_pil,
                        ref_notes,
                        arm=arm,
                        use_bf16=use_bf16,
                        device=device,
                        eval_s31=eval_s31,
                        rlvr_grpo=rlvr_grpo,
                    )
                rows.append({
                    "piece": slug,
                    "handwritten": is_hw,
                    "domain": domain,
                    "n_ref_notes": len(ref_notes),
                    "arms": arm_res,
                })
                done_keys.add((slug, domain))
                b_f1 = arm_res["baseline_1280"]["note_f1_py"]
                print(
                    f"[{idx:02d}/{len(all_pieces):02d}] {slug:32s} [{domain:5s}] base={b_f1:.4f} | "
                    + " ".join(f"{k}:{arm_res[k]['note_f1_py'] - b_f1:+.4f}" for k in arm_keys[1:]),
                    flush=True,
                )
                if (time.perf_counter() - last_save) >= SAVE_EVERY_MINUTES * 60.0 or domain == "photo":
                    _atomic_write_json(
                        partial_path,
                        {
                            "exp_id": EXP_ID,
                            "status": "RUNNING",
                            "summary": _summarize(rows, arm_keys),
                            "rows": rows,
                        },
                    )
                    last_save = time.perf_counter()

        summary = _summarize(rows, arm_keys)
        elapsed_secs = round(time.perf_counter() - t_start, 2)
        final_report = {
            "exp_id": EXP_ID,
            "status": "COMPLETED",
            "started_utc": started_utc,
            "completed_utc": _utc_now(),
            "elapsed_secs": elapsed_secs,
            "platform": platform_str,
            "base_ckpt": HF_CKPT_PATH,
            "base_ckpt_meta": ckpt_meta,
            "config": {
                "arms": ARMS,
                "base_max_dim": BASE_MAX_DIM,
                "unitsize_dim_range": [MIN_UNITSIZE_DIM, MAX_UNITSIZE_DIM],
                "target_unit_size_px": TARGET_UNIT_SIZE_PX,
                "max_len": MAX_LEN,
            },
            "summary": summary,
            "rows": rows,
        }
        final_local = CACHE_ROOT / "evals" / f"{EXP_ID}.json"
        _atomic_write_json(final_local, final_report)
        api.upload_file(
            path_or_fileobj=str(final_local),
            path_in_repo=HF_EVAL_PATH,
            repo_id=HF_REPO,
            repo_type="dataset",
            commit_message=f"{EXP_ID}: upload OEMER dewarp + Homr unit_size eval JSON",
        )
        ledger_local = CACHE_ROOT / "runs" / f"{EXP_ID}.json"
        _atomic_write_json(
            ledger_local,
            {
                "exp_id": EXP_ID,
                "status": "COMPLETED",
                "started_utc": started_utc,
                "completed_utc": final_report["completed_utc"],
                "elapsed_secs": elapsed_secs,
                "platform": platform_str,
                "hf_eval_path": HF_EVAL_PATH,
                "summary": summary,
            },
        )
        api.upload_file(
            path_or_fileobj=str(ledger_local),
            path_in_repo=HF_RUN_PATH,
            repo_id=HF_REPO,
            repo_type="dataset",
            commit_message=f"{EXP_ID}: mark COMPLETED",
        )
        print(f"[{EXP_ID}] COMPLETED in {elapsed_secs}s", flush=True)
        return 0
    finally:
        if TOKEN_PATH.exists():
            try:
                TOKEN_PATH.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
