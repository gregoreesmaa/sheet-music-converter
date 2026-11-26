#!/usr/bin/env python3
"""Google Colab L4 runner for i13-14: 2026-11-26T07-40-i13-14-photo-illum-flatten-autocrop.

Commercial OMR Insight (Halbestunde, PlayScore 2, Newzik, Soundslice):
  In ov-36, naive CLAHE amplified paper grain and Hough deskew clamped on dark
  desk borders, hurting hw_photo. Commercial mobile sheet-music scanners instead
  use:
    1. Low-frequency Gaussian background division (strips broad phone shadows
       and yellow paper gradients without boosting high-frequency paper grain)
    2. Safe 4-edge dark-desk auto-crop (reclaims 10-25% effective resolution
       wasted on table borders while preserving >= 2% quiet safety margin;
       guaranteed no-op on clean white pages).

Tests 5 isolated arms on all 45 suite31_hw pieces (clean + photo = 90 pages)
at canonical 1280px / 3840:
  1. baseline
  2. bg_divide_soft_a50
  3. bg_divide_full_a85
  4. desk_autocrop_safe
  5. bg_soft_plus_autocrop

Expects on Colab VM:
  - `/content/bundle.tar.gz` (contains `sheet-music-converter/...` and `mscz-rs/`)
  - `/tmp/.hf_token` (Hugging Face token; chmod 600, deleted immediately and in `finally:`)
"""

from __future__ import annotations

import contextlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

EXP_ID = "2026-11-26T07-40-i13-14-photo-illum-flatten-autocrop"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_SUITE31_TAR = "datasets/suite31_hw_eval.tar"
HF_CKPT_PATH = os.environ.get("HF_CKPT_PATH", "checkpoints/iter10_soup_baseline/uni_omr_soup.pt")
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

BUNDLE_PATH = Path("/content/bundle.tar.gz")
WORK_ROOT = Path("/content/workspace")
CACHE_ROOT = Path("/content/cache")
HF_TOKEN_FILE = Path("/tmp/.hf_token")

MAX_DIM = 1280
MAX_LEN = 3840
SAVE_EVERY_MINUTES = 5.0

SWEEP_ARMS: List[Dict[str, Any]] = [
    {"key": "baseline", "bg_alpha": 0.0, "autocrop": False},
    {"key": "bg_divide_soft_a50", "bg_alpha": 0.50, "autocrop": False},
    {"key": "bg_divide_full_a85", "bg_alpha": 0.85, "autocrop": False},
    {"key": "desk_autocrop_safe", "bg_alpha": 0.0, "autocrop": True},
    {"key": "bg_soft_plus_autocrop", "bg_alpha": 0.50, "autocrop": True},
]


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _run_cmd(cmd: Sequence[str] | str, *, shell: bool = False, env: Optional[Dict[str, str]] = None) -> None:
    print(f"[{EXP_ID}] $ {cmd if isinstance(cmd, str) else ' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd, shell=shell, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed (rc={rc}): {cmd}")


def _read_and_purge_hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token and HF_TOKEN_FILE.exists():
        try:
            os.chmod(HF_TOKEN_FILE, 0o600)
        except OSError:
            pass
        token = HF_TOKEN_FILE.read_text(encoding="utf-8").strip()
    if HF_TOKEN_FILE.exists():
        try:
            HF_TOKEN_FILE.unlink()
        except OSError:
            pass
    if not token:
        raise RuntimeError("HF token not found in /tmp/.hf_token or HF_TOKEN env var")
    return token


def _find_repo_root() -> Path:
    for cand in (
        WORK_ROOT / "sheet-music-converter",
        Path("/workspace/sheet-music-converter"),
        Path("/content/sheet-music-converter"),
    ):
        if (cand / "tmp-train" / "mvd" / "eval_uni_suite31.py").exists():
            return cand
    if BUNDLE_PATH.exists():
        WORK_ROOT.mkdir(parents=True, exist_ok=True)
        with tarfile.open(BUNDLE_PATH, "r:*") as tf:
            tf.extractall(path=WORK_ROOT)
        repo = WORK_ROOT / "sheet-music-converter"
        if (repo / "tmp-train" / "mvd" / "eval_uni_suite31.py").exists():
            return repo
    raise FileNotFoundError(f"Could not locate sheet-music-converter workspace or bundle at {BUNDLE_PATH}")


def _setup_environment() -> Tuple[Path, Path, Path]:
    repo_root = _find_repo_root()
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    needed_pkgs = []
    for mod, pkg in [("huggingface_hub", "huggingface_hub>=0.26"), ("PIL", "pillow"), ("numpy", "numpy")]:
        try:
            __import__(mod)
        except ImportError:
            needed_pkgs.append(pkg)
    if needed_pkgs:
        _run_cmd([sys.executable, "-m", "pip", "install", "-q", *needed_pkgs])

    cargo_bin_dir = Path.home() / ".cargo" / "bin"
    env = os.environ.copy()
    env["PATH"] = f"{cargo_bin_dir}:{env.get('PATH', '')}"

    aser_manifest = repo_root / "converter" / "assembly-serialize" / "Cargo.toml"
    eharness_manifest = repo_root / "converter" / "eval-harness" / "Cargo.toml"
    aser_bin = aser_manifest.parent / "target" / "release" / "assembly-serialize"
    eharness_bin = eharness_manifest.parent / "target" / "release" / "eval-harness"

    if not aser_bin.exists() or not eharness_bin.exists():
        if shutil.which("cargo", path=env["PATH"]) is None:
            _run_cmd(
                "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
                "| sh -s -- -y --default-toolchain stable --profile minimal",
                shell=True,
                env=env,
            )
        if not aser_bin.exists():
            _run_cmd(["cargo", "build", "--release", "--manifest-path", str(aser_manifest)], env=env)
        if not eharness_bin.exists():
            _run_cmd(["cargo", "build", "--release", "--manifest-path", str(eharness_manifest)], env=env)

    assert aser_bin.exists(), f"Missing Linux binary: {aser_bin}"
    assert eharness_bin.exists(), f"Missing Linux binary: {eharness_bin}"
    return repo_root, aser_bin, eharness_bin


def _safe_desk_autocrop(pil_img: Any, *, max_trim_frac: float = 0.18, pad_frac: float = 0.02) -> Tuple[Any, bool]:
    """Trim dark desk borders from all 4 edges with >= 2% quiet safety margin.

    Guaranteed no-op on clean white-margin pages.
    """
    import numpy as np

    rgb = pil_img.convert("RGB")
    arr = np.asarray(rgb, dtype=np.float32)
    h, w = arr.shape[:2]
    luma = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]

    max_y = int(h * max_trim_frac)
    max_x = int(w * max_trim_frac)

    def _is_dark_band(vec: Any) -> bool:
        return bool(float(vec.mean()) < 85.0 or float((vec < 90.0).mean()) >= 0.65)

    top = 0
    while top < max_y and _is_dark_band(luma[top, :]):
        top += 1
    bot = h - 1
    while (h - 1 - bot) < max_y and _is_dark_band(luma[bot, :]):
        bot -= 1
    left = 0
    while left < max_x and _is_dark_band(luma[:, left]):
        left += 1
    right = w - 1
    while (w - 1 - right) < max_x and _is_dark_band(luma[:, right]):
        right -= 1

    if top < int(0.01 * h) and (h - 1 - bot) < int(0.01 * h) and left < int(0.01 * w) and (w - 1 - right) < int(0.01 * w):
        return rgb, False

    pad_y = max(4, int(round(h * pad_frac)))
    pad_x = max(4, int(round(w * pad_frac)))
    y0 = max(0, top - pad_y)
    y1 = min(h, bot + 1 + pad_y)
    x0 = max(0, left - pad_x)
    x1 = min(w, right + 1 + pad_x)
    if (y1 - y0) < int(0.6 * h) or (x1 - x0) < int(0.6 * w):
        return rgb, False
    return rgb.crop((x0, y0, x1, y1)), True


def _bg_divide_flatten(pil_img: Any, alpha: float, sigma_frac: float = 0.04) -> Any:
    """Low-frequency Gaussian background division (grain-safe shadow/yellowing removal)."""
    import numpy as np
    from PIL import Image, ImageFilter

    rgb = pil_img.convert("RGB")
    if alpha <= 0.0:
        return rgb
    w, h = rgb.size
    small_w = max(32, w // 4)
    small_h = max(32, h // 4)
    radius = max(8, int(round(max(small_w, small_h) * sigma_frac)))
    bg_small = rgb.resize((small_w, small_h), Image.BILINEAR).filter(ImageFilter.GaussianBlur(radius=radius))
    bg_full = bg_small.resize((w, h), Image.BILINEAR)

    img_f = np.asarray(rgb, dtype=np.float32)
    bg_f = np.maximum(np.asarray(bg_full, dtype=np.float32), 32.0)
    white_ref = float(np.percentile(bg_f, 98))
    white_ref = max(180.0, min(255.0, white_ref))
    flat = np.clip((img_f / bg_f) * white_ref, 0.0, 255.0)
    blended = np.clip((1.0 - alpha) * img_f + alpha * flat, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(blended, mode="RGB")


def _summarize_arm(piece_rows: List[Dict[str, Any]], base_total_secs: Optional[float]) -> Dict[str, Any]:
    slices = {
        "printed_clean": [r for r in piece_rows if not r["handwritten"]],
        "printed_photo": [r for r in piece_rows if not r["handwritten"]],
        "hw_clean": [r for r in piece_rows if r["handwritten"]],
        "hw_photo": [r for r in piece_rows if r["handwritten"]],
        "all45_clean": piece_rows,
        "all45_photo": piece_rows,
    }
    out: Dict[str, Any] = {}
    for sname, rows in slices.items():
        dom = "clean" if sname.endswith("clean") else "photo"
        nf1 = [float(r[f"{dom}_note_f1"]) for r in rows]
        tf1 = [float(r[f"{dom}_tok_f1"]) for r in rows]
        dec = [1.0 if r[f"{dom}_decode_ok"] else 0.0 for r in rows]
        out[sname] = {
            "n": len(rows),
            "note_f1_mean": round(statistics.fmean(nf1), 6) if nf1 else 0.0,
            "tok_f1_mean": round(statistics.fmean(tf1), 6) if tf1 else 0.0,
            "decode_ok_rate": round(statistics.fmean(dec), 6) if dec else 0.0,
        }
    total_secs = sum(float(r["clean_elapsed_secs"]) + float(r["photo_elapsed_secs"]) for r in piece_rows)
    overhead = ((total_secs - base_total_secs) / max(1e-6, base_total_secs) * 100.0) if base_total_secs else 0.0
    out["runtime"] = {
        "total_decode_secs_90pages": round(total_secs, 2),
        "overhead_vs_baseline_pct": round(overhead, 2),
    }
    return out


def run_experiment() -> Dict[str, Any]:
    started_ts = time.time()
    started_iso = datetime.now(timezone.utc).isoformat()
    try:
        token = _read_and_purge_hf_token()
        repo_root, aser_bin, eharness_bin = _setup_environment()

        mvd_dir = repo_root / "tmp-train" / "mvd"
        if str(mvd_dir) not in sys.path:
            sys.path.insert(0, str(mvd_dir))

        from PIL import Image
        import torch
        from huggingface_hub import HfApi, hf_hub_download
        import eval_sys_gate as esg
        import eval_uni_suite31 as ev
        import uni_omr

        esg.ASER = str(aser_bin)
        esg.EHARNESS = str(eharness_bin)
        if torch.cuda.is_available():
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_flash_sdp(True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_bf16 = device.type == "cuda"
        platform_str = f"colab-{torch.cuda.get_device_name(0)}" if device.type == "cuda" else "cpu"

        api = HfApi(token=token)
        ckpt_local = Path(
            hf_hub_download(
                repo_id=HF_REPO_ID, filename=HF_CKPT_PATH, repo_type="dataset",
                token=token, local_dir=str(CACHE_ROOT / "hf"),
            )
        )
        suite_tar = Path(
            hf_hub_download(
                repo_id=HF_REPO_ID, filename=HF_SUITE31_TAR, repo_type="dataset",
                token=token, local_dir=str(CACHE_ROOT / "hf"),
            )
        )
        suite_dir = CACHE_ROOT / "suite31_eval"
        if not (suite_dir / "suite31" / "suite31_hw_manifest.json").exists():
            suite_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(suite_tar, "r:*") as tf:
                tf.extractall(path=suite_dir)

        manifest_path = suite_dir / "suite31" / "suite31_hw_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        pieces = manifest["pieces"]
        suite_root = manifest_path.parent

        model, cfg, ckpt_meta = ev.load_uni_checkpoint(ckpt_local, device)
        _, _, eos_id, _, _, _ = ev._vocab_spec(int(cfg.vocab_size))

        workdir = CACHE_ROOT / EXP_ID
        workdir.mkdir(parents=True, exist_ok=True)
        partial_path = workdir / f"{EXP_ID}_partial.json"
        final_local_path = workdir / f"{EXP_ID}.json"

        refs_cache: Dict[str, Tuple[List[str], Path]] = {}
        for piece in pieces:
            slug = piece["slug"]
            ref_bek = suite_root / piece["bekern"]
            ref_toks = ref_bek.read_text(encoding="utf-8").strip().split()
            ref_notes = workdir / f"ref-{slug}.json"
            if not ref_notes.exists():
                esg.run_cli([esg.ASER, "bekern-decode", "--input", str(ref_bek), "--out", str(ref_notes)])
            refs_cache[slug] = (ref_toks, ref_notes)

        results_by_arm: Dict[str, Any] = {}
        if partial_path.exists():
            try:
                results_by_arm = json.loads(partial_path.read_text(encoding="utf-8")).get("arms", {})
            except Exception:
                results_by_arm = {}

        base_total_secs: Optional[float] = None
        if "baseline" in results_by_arm:
            base_total_secs = results_by_arm["baseline"]["summary"]["runtime"]["total_decode_secs_90pages"]
        base_hyp_cache: Dict[Tuple[str, str], Tuple[List[str], Dict[str, Any]]] = {}
        last_save_ts = time.time()

        def _build_report(status: str) -> Dict[str, Any]:
            comp = {
                k: {
                    s: v["summary"][s]["note_f1_mean"]
                    for s in ("printed_clean", "printed_photo", "hw_clean", "hw_photo", "all45_clean", "all45_photo")
                } | {"overhead_pct": v["summary"]["runtime"]["overhead_vs_baseline_pct"]}
                for k, v in results_by_arm.items()
            }
            return {
                "exp_id": EXP_ID,
                "status": status,
                "started_utc": started_iso,
                "elapsed_secs": round(time.time() - started_ts, 2),
                "platform": platform_str,
                "ckpt_hf_path": HF_CKPT_PATH,
                "ckpt_meta": ckpt_meta,
                "comparison_table": comp,
                "arms": results_by_arm,
            }

        for arm in SWEEP_ARMS:
            akey = arm["key"]
            if akey in results_by_arm:
                print(f"[{EXP_ID}] Skipping completed arm: {akey}", flush=True)
                continue

            print(f"[{EXP_ID}] Running arm: {akey} ({arm})", flush=True)
            piece_rows: List[Dict[str, Any]] = []
            n_cropped_clean, n_cropped_photo = 0, 0
            for piece in pieces:
                slug = piece["slug"]
                ref_toks, ref_notes = refs_cache[slug]
                prow: Dict[str, Any] = {"piece": slug, "handwritten": bool(piece.get("handwritten", False))}
                for domain in ("clean", "photo"):
                    src_path = suite_root / piece[domain]
                    t0 = time.perf_counter()
                    use_path = src_path
                    cropped = False
                    if arm["autocrop"] or arm["bg_alpha"] > 0.0:
                        with Image.open(src_path) as im:
                            proc_im = im.convert("RGB")
                            if arm["autocrop"]:
                                proc_im, cropped = _safe_desk_autocrop(proc_im)
                            if arm["bg_alpha"] > 0.0:
                                proc_im = _bg_divide_flatten(proc_im, alpha=float(arm["bg_alpha"]))
                            if cropped or arm["bg_alpha"] > 0.0:
                                tmp_png = workdir / f"prep-{akey}-{slug}-{domain}.png"
                                proc_im.save(tmp_png)
                                use_path = tmp_png
                    if cropped:
                        if domain == "clean":
                            n_cropped_clean += 1
                        else:
                            n_cropped_photo += 1

                    # If image was untouched and baseline is cached, reuse baseline directly!
                    if use_path == src_path and akey != "baseline" and (slug, domain) in base_hyp_cache:
                        _, scores = base_hyp_cache[(slug, domain)]
                        elapsed = time.perf_counter() - t0
                    else:
                        img_t = uni_omr.load_fullpage_image(use_path, max_dim=MAX_DIM).unsqueeze(0).to(device)
                        ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else contextlib.nullcontext()
                        with torch.no_grad(), ctx:
                            seq, _ = ev.generate_mtp_spec(
                                model, img_t, max_len=MAX_LEN, use_grammar=True, strict_grammar=False, return_stats=True
                            )
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                        elapsed = max(1e-6, time.perf_counter() - t0)

                        seq_ids = seq[0].detach().cpu().tolist()
                        pred_raw = ev.decode_uni_compat(seq_ids, vocab_size=int(cfg.vocab_size), skip_special=True)
                        if eos_id not in seq_ids[1:]:
                            pred_raw = ev.trim_truncated_tokens(pred_raw)
                        hyp_toks, _ = ev.stitch_uni_tokens(pred_raw)

                        cached = base_hyp_cache.get((slug, domain))
                        if cached is not None and hyp_toks == cached[0]:
                            scores = dict(cached[1])
                        else:
                            _, _, tok_f1 = uni_omr.compute_token_f1(ref_toks, hyp_toks)
                            hyp_txt = workdir / f"{akey}-{slug}-{domain}-hyp.txt"
                            hyp_txt.write_text(" ".join(hyp_toks) + "\n", encoding="utf-8")
                            hyp_notes = workdir / f"{akey}-{slug}-{domain}-notes.json"
                            p_dec = esg.run_cli([esg.ASER, "bekern-decode", "--input", str(hyp_txt), "--out", str(hyp_notes)])
                            bekern_ok = (p_dec.returncode == 0) and hyp_notes.exists()
                            if not bekern_ok:
                                scores = {"tok_f1": round(tok_f1, 6), "note_f1": 0.0, "decode_ok": False}
                            else:
                                sc = esg.score_notes_pair(ref_notes, hyp_notes, workdir, f"{akey}-{slug}-{domain}", esg.EHARNESS, esg.ASER)
                                scores = {
                                    "tok_f1": round(tok_f1, 6),
                                    "note_f1": round(float(sc.get("f1", 0.0)), 6),
                                    "decode_ok": not bool(sc.get("zeroed")),
                                }
                            if akey == "baseline":
                                base_hyp_cache[(slug, domain)] = (hyp_toks, dict(scores))

                    prow[f"{domain}_tok_f1"] = scores["tok_f1"]
                    prow[f"{domain}_note_f1"] = scores["note_f1"]
                    prow[f"{domain}_decode_ok"] = scores["decode_ok"]
                    prow[f"{domain}_cropped"] = cropped
                    prow[f"{domain}_elapsed_secs"] = round(elapsed, 4)

                piece_rows.append(prow)
                if (time.time() - last_save_ts) >= SAVE_EVERY_MINUTES * 60.0:
                    _atomic_write_json(partial_path, _build_report("RUNNING"))
                    last_save_ts = time.time()

            arm_summary = _summarize_arm(piece_rows, base_total_secs)
            arm_summary["autocrop_counts"] = {"clean": n_cropped_clean, "photo": n_cropped_photo}
            if akey == "baseline":
                base_total_secs = arm_summary["runtime"]["total_decode_secs_90pages"]
            results_by_arm[akey] = {"spec": arm, "summary": arm_summary, "pieces": piece_rows}
            print(
                f"[{EXP_ID}] DONE {akey}: all45_clean={arm_summary['all45_clean']['note_f1_mean']:.4f} "
                f"all45_photo={arm_summary['all45_photo']['note_f1_mean']:.4f} "
                f"hw_photo={arm_summary['hw_photo']['note_f1_mean']:.4f} "
                f"cropped=(clean:{n_cropped_clean}, photo:{n_cropped_photo})",
                flush=True,
            )
            _atomic_write_json(partial_path, _build_report("RUNNING"))
            last_save_ts = time.time()

        final_report = _build_report("COMPLETED")
        _atomic_write_json(final_local_path, final_report)
        _atomic_write_json(partial_path, final_report)

        api.upload_file(
            path_or_fileobj=str(final_local_path),
            path_in_repo=HF_EVAL_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"{EXP_ID}: upload grain-safe illumination flatten + autocrop JSON (Colab L4)",
        )
        ledger = {
            "exp_id": EXP_ID,
            "status": "COMPLETED",
            "started_utc": started_iso,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_secs": final_report["elapsed_secs"],
            "platform": platform_str,
            "hf_eval_path": HF_EVAL_PATH,
            "ckpt_hf_path": HF_CKPT_PATH,
            "comparison_table": final_report["comparison_table"],
        }
        tmp_ledger = workdir / "run_ledger_final.json"
        _atomic_write_json(tmp_ledger, ledger)
        api.upload_file(
            path_or_fileobj=str(tmp_ledger),
            path_in_repo=HF_RUN_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"{EXP_ID}: mark COMPLETED (Colab L4)",
        )
        print(f"[{EXP_ID}] Uploaded HF artifacts: {HF_EVAL_PATH} & {HF_RUN_PATH}", flush=True)
        return final_report
    finally:
        if HF_TOKEN_FILE.exists():
            try:
                HF_TOKEN_FILE.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    run_experiment()
