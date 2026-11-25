#!/usr/bin/env python3
"""Google Colab L4 runner for Experiment 6: 2026-11-26T00-55-trial-scribe-ttt-suite31-hw.

Tests SCRIBE / DocTTT per-manuscript Test-Time Training (`trial_2028_ttt_adapt.py`)
on ALL 14 unseen historical handwritten manuscripts in `suite31_hw` (`clean` +
`photo` = 28 pages), starting from `checkpoints/iter10_soup_baseline/uni_omr_soup.pt`.

Evaluates both `ttt_steps=3` and `ttt_steps=5` against the exact same pre-TTT
baseline decode (resetting adapted RMSNorm weights bit-exact `atol=0` between
pieces/domains), checkpoints progress locally every <= 5 minutes, and uploads:
  - `evals/2026-11-26T00-55-trial-scribe-ttt-suite31-hw.json`
  - `runs/2026-11-26T00-55-trial-scribe-ttt-suite31-hw.json`
to Hugging Face Private Dataset Hub (`gregoreesmaa/sheet-music-converter-omr`).

Expected pre-uploaded files on Colab VM:
  - `/content/exp6_bundle.tar.gz` (contains `mscz-rs/`, `sheet-music-converter/...`)
  - `/tmp/.hf_token` (deleted immediately after reading and in `finally:`)
"""

from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

EXP_ID = "2026-11-26T00-55-trial-scribe-ttt-suite31-hw"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_SUITE31_TAR = "datasets/suite31_hw_eval.tar"
HF_CKPT_PATH = "checkpoints/iter10_soup_baseline/uni_omr_soup.pt"
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

BUNDLE_PATH = Path("/content/exp6_bundle.tar.gz")
WORKSPACE_ROOT = Path("/workspace")
VOL_ROOT = Path("/content/vol")
HF_TOKEN_FILE = Path("/tmp/.hf_token")


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _run_cmd(cmd: Sequence[str] | str, *, shell: bool = False, env: Optional[Dict[str, str]] = None) -> None:
    print(f"[{EXP_ID}] $ {cmd if isinstance(cmd, str) else ' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd,
        shell=shell,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed (rc={rc}): {cmd}")


def _mean_metric(rows: Sequence[Dict[str, Any]], section: str, key: str) -> Optional[float]:
    vals = [
        float(r[section][key])
        for r in rows
        if isinstance(r.get(section), dict) and r[section].get(key) is not None
    ]
    return round(sum(vals) / len(vals), 6) if vals else None


def _summarize_rows(rows: Sequence[Dict[str, Any]], domains: Sequence[str], step_counts: Sequence[int]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for domain in domains:
        drows = [r for r in rows if r["domain"] == domain]
        bef_nf1 = _mean_metric(drows, "before", "note_f1")
        bef_tf1 = _mean_metric(drows, "before", "tok_f1")
        dom_sum: Dict[str, Any] = {
            "n": len(drows),
            "before_note_f1": bef_nf1,
            "before_tok_f1": bef_tf1,
            "steps": {},
        }
        for s in step_counts:
            key_after = f"after_s{s}"
            key_stats = f"ttt_stats_s{s}"
            aft_nf1 = _mean_metric(drows, key_after, "note_f1")
            aft_tf1 = _mean_metric(drows, key_after, "tok_f1")
            times = [
                float(r[key_stats]["elapsed_secs"])
                for r in drows
                if isinstance(r.get(key_stats), dict) and r[key_stats].get("elapsed_secs") is not None
            ]
            wins = sum(
                1
                for r in drows
                if isinstance(r.get(key_after), dict)
                and float(r[key_after]["note_f1"]) > float(r["before"]["note_f1"]) + 1e-6
            )
            losses = sum(
                1
                for r in drows
                if isinstance(r.get(key_after), dict)
                and float(r[key_after]["note_f1"]) < float(r["before"]["note_f1"]) - 1e-6
            )
            ties = len(drows) - wins - losses
            dom_sum["steps"][str(s)] = {
                "after_note_f1": aft_nf1,
                "delta_note_f1": (
                    round(aft_nf1 - bef_nf1, 6)
                    if (bef_nf1 is not None and aft_nf1 is not None)
                    else None
                ),
                "after_tok_f1": aft_tf1,
                "delta_tok_f1": (
                    round(aft_tf1 - bef_tf1, 6)
                    if (bef_tf1 is not None and aft_tf1 is not None)
                    else None
                ),
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "median_ttt_secs": round(float(statistics.median(times)), 4) if times else None,
            }
        summary[domain] = dom_sum
    return summary


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


def _setup_environment() -> tuple[Path, Path]:
    # 1. Extract uploaded source bundle to /workspace
    if not (WORKSPACE_ROOT / "sheet-music-converter" / "tmp-train" / "mvd").exists():
        if not BUNDLE_PATH.exists():
            raise FileNotFoundError(f"Missing uploaded bundle: {BUNDLE_PATH}")
        WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
        print(f"[{EXP_ID}] Extracting {BUNDLE_PATH} -> {WORKSPACE_ROOT}...", flush=True)
        with tarfile.open(BUNDLE_PATH, "r:*") as tf:
            tf.extractall(path=WORKSPACE_ROOT)
        print(f"[{EXP_ID}] Extracted bundle cleanly.", flush=True)

    # 2. Ensure lightweight Python packages (torch is pre-installed on Colab)
    needed_pkgs = []
    for mod, pkg in [
        ("huggingface_hub", "huggingface_hub>=0.26"),
        ("scipy", "scipy"),
        ("PIL", "pillow"),
        ("numpy", "numpy"),
        ("tqdm", "tqdm"),
    ]:
        try:
            __import__(mod)
        except ImportError:
            needed_pkgs.append(pkg)
    if needed_pkgs:
        _run_cmd([sys.executable, "-m", "pip", "install", "-q", *needed_pkgs])

    # 3. Ensure Rust toolchain + build assembly-serialize & eval-harness
    cargo_bin_dir = Path.home() / ".cargo" / "bin"
    env = os.environ.copy()
    env["PATH"] = f"{cargo_bin_dir}:{env.get('PATH', '')}"
    if shutil.which("cargo", path=env["PATH"]) is None:
        print(f"[{EXP_ID}] Installing minimal Rust toolchain via rustup...", flush=True)
        _run_cmd(
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
            "| sh -s -- -y --default-toolchain stable --profile minimal",
            shell=True,
            env=env,
        )

    aser_manifest = WORKSPACE_ROOT / "sheet-music-converter" / "converter" / "assembly-serialize" / "Cargo.toml"
    eharness_manifest = WORKSPACE_ROOT / "sheet-music-converter" / "converter" / "eval-harness" / "Cargo.toml"
    aser_bin = aser_manifest.parent / "target" / "release" / "assembly-serialize"
    eharness_bin = eharness_manifest.parent / "target" / "release" / "eval-harness"

    if not aser_bin.exists():
        _run_cmd(["cargo", "build", "--release", "--manifest-path", str(aser_manifest)], env=env)
    if not eharness_bin.exists():
        _run_cmd(["cargo", "build", "--release", "--manifest-path", str(eharness_manifest)], env=env)

    assert aser_bin.exists(), f"Missing Linux binary: {aser_bin}"
    assert eharness_bin.exists(), f"Missing Linux binary: {eharness_bin}"
    return aser_bin, eharness_bin


def run_ttt_experiment(
    step_counts: List[int] = [3, 5],
    ttt_lr: float = 5e-5,
    adapt_scope: str = "norms",
    conf_threshold: float = 0.85,
    cons_weight: float = 1.0,
    ent_weight: float = 1.0,
    kl_weight: float = 0.5,
    l2_weight: float = 1e-2,
    grad_clip: float = 1.0,
    max_dim: int = 1536,
    max_len: int = 4096,
    ttt_max_decode_len: int = 512,
    save_every_minutes: float = 5.0,
) -> Dict[str, Any]:
    ttt = None
    try:
        hf_token = _read_and_purge_hf_token()
        aser_bin, eharness_bin = _setup_environment()

        import torch
        from PIL import Image
        from huggingface_hub import HfApi, hf_hub_download

        t_start = time.time()
        started_iso = datetime.now(timezone.utc).isoformat()

        # SDPA rule: avoid cuDNN plan recompilation on variable (mem, T) shapes
        if torch.cuda.is_available():
            torch.backends.cuda.enable_cudnn_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_flash_sdp(True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        cc = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
        use_bf16 = bool(torch.cuda.is_available() and cc[0] >= 8)
        platform_str = f"colab-{gpu_name}"
        print(
            f"[{EXP_ID}] Starting on {platform_str} (cc={cc[0]}.{cc[1]}, bf16={use_bf16}, "
            f"torch={torch.__version__}) | step_counts={step_counts} lr={ttt_lr:g} "
            f"scope={adapt_scope} max_dim={max_dim} max_len={max_len}",
            flush=True,
        )

        api = HfApi(token=hf_token)

        # 1. Download & unpack suite31_hw_eval dataset
        VOL_ROOT.mkdir(parents=True, exist_ok=True)
        suite31_dir = VOL_ROOT / "datasets" / "suite31_hw_eval"
        manifest_path = suite31_dir / "suite31" / "suite31_hw_manifest.json"
        if not manifest_path.exists():
            print(f"[{EXP_ID}] Downloading {HF_SUITE31_TAR} from HF...", flush=True)
            tar_local = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=HF_SUITE31_TAR,
                repo_type="dataset",
                token=hf_token,
                local_dir=str(VOL_ROOT),
            )
            suite31_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tar_local, "r:*") as tf:
                tf.extractall(path=suite31_dir)
            print(f"[{EXP_ID}] Extracted suite31_hw_eval -> {suite31_dir}", flush=True)
        else:
            print(f"[{EXP_ID}] Using cached suite31_hw_eval at {manifest_path}", flush=True)

        # 2. Download iter10_soup_baseline checkpoint
        ckpt_path = VOL_ROOT / HF_CKPT_PATH
        if not ckpt_path.exists():
            print(f"[{EXP_ID}] Downloading {HF_CKPT_PATH} from HF...", flush=True)
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=HF_CKPT_PATH,
                repo_type="dataset",
                token=hf_token,
                local_dir=str(VOL_ROOT),
            )
            print(f"[{EXP_ID}] Cached checkpoint -> {ckpt_path}", flush=True)
        else:
            print(f"[{EXP_ID}] Using cached checkpoint at {ckpt_path}", flush=True)

        # 3. Wire Linux Rust binaries & import UniOMR / TTT modules
        mvd_dir = str(WORKSPACE_ROOT / "sheet-music-converter" / "tmp-train" / "mvd")
        if mvd_dir not in sys.path:
            sys.path.insert(0, mvd_dir)
        import eval_sys_gate as esg
        import eval_uni_suite31 as ev
        import trial_2028_ttt_adapt as ttt_mod
        import uni_omr

        esg.ASER = aser_bin
        esg.EHARNESS = eharness_bin
        print(f"[{EXP_ID}] Wired Linux Rust binaries: {aser_bin} & {eharness_bin}", flush=True)

        # 4. Load model and resolve all 14 real historical HW pieces
        model, cfg, ckpt_meta = ev.load_uni_checkpoint(ckpt_path, device=device)
        _, all_pieces = ev.resolve_manifest_pieces(manifest_path)
        hw_pieces = [p for p in all_pieces if p["handwritten"]]
        domains = ["clean", "photo"]
        print(
            f"[{EXP_ID}] Loaded ckpt step={ckpt_meta.get('step')} | "
            f"HW pieces={len(hw_pieces)} | domains={domains}",
            flush=True,
        )

        ttt = ttt_mod.TTTSession(
            model=model,
            ttt_lr=ttt_lr,
            adapt_scope=adapt_scope,
            conf_threshold=conf_threshold,
            cons_weight=cons_weight,
            ent_weight=ent_weight,
            kl_weight=kl_weight,
            l2_weight=l2_weight,
            grad_clip=grad_clip,
            use_bf16=use_bf16,
            strict_grammar=False,
            max_decode_len=ttt_max_decode_len,
        )
        print(
            f"[{EXP_ID}] TTTSession ready: scope={ttt.adapt_scope}, "
            f"adapted_params={ttt.adapted_param_count:,} ({ttt.adapted_param_fraction * 100:.4f}%)",
            flush=True,
        )

        # 5. Resume support from local partial checkpoint if present
        partial_path = VOL_ROOT / "evals" / f"{EXP_ID}_partial.json"
        final_local_path = VOL_ROOT / "evals" / f"{EXP_ID}.json"
        rows: List[Dict[str, Any]] = []
        done_keys: set[tuple[str, str]] = set()
        if partial_path.exists():
            try:
                prev = json.loads(partial_path.read_text(encoding="utf-8"))
                if isinstance(prev.get("rows"), list):
                    rows = prev["rows"]
                    done_keys = {(r["piece"], r["domain"]) for r in rows}
                    print(
                        f"[{EXP_ID}] Resuming from {partial_path}: {len(done_keys)}/28 pairs already done",
                        flush=True,
                    )
            except Exception as exc:
                print(f"[{EXP_ID}] Ignoring unreadable partial file: {exc}", flush=True)

        workdir = Path(tempfile.mkdtemp(prefix="exp6_ttt_"))
        last_save_ts = time.time()
        save_interval_s = max(30.0, float(save_every_minutes) * 60.0)

        def _build_report(status: str) -> Dict[str, Any]:
            return {
                "exp_id": EXP_ID,
                "status": status,
                "started_utc": started_iso,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_secs": round(time.time() - t_start, 2),
                "platform": platform_str,
                "bf16": use_bf16,
                "ckpt_hf_path": HF_CKPT_PATH,
                "ckpt_meta": ckpt_meta,
                "model_config": asdict(cfg),
                "eval_config": {
                    "max_dim": max_dim,
                    "max_len": max_len,
                    "use_grammar": True,
                    "strict_grammar": False,
                    "use_mtp_spec": True,
                    "domains": domains,
                    "n_hw_pieces": len(hw_pieces),
                },
                "ttt_config": {
                    "step_counts": list(step_counts),
                    "ttt_lr": ttt_lr,
                    "adapt_scope": adapt_scope,
                    "adapted_param_count": ttt.adapted_param_count,
                    "adapted_param_fraction": round(ttt.adapted_param_fraction, 6),
                    "conf_threshold": conf_threshold,
                    "cons_weight": cons_weight,
                    "ent_weight": ent_weight,
                    "kl_weight": kl_weight,
                    "l2_weight": l2_weight,
                    "grad_clip": grad_clip,
                    "ttt_max_decode_len": ttt_max_decode_len,
                },
                "summary": _summarize_rows(rows, domains, step_counts),
                "rows": rows,
            }

        # Upload initial RUNNING ledger to HF
        try:
            run_ledger = {
                "exp_id": EXP_ID,
                "status": "RUNNING",
                "started_utc": started_iso,
                "platform": platform_str,
                "hf_eval_path": HF_EVAL_PATH,
                "ckpt_hf_path": HF_CKPT_PATH,
                "step_counts": list(step_counts),
            }
            tmp_run = workdir / "run_ledger.json"
            _atomic_write_json(tmp_run, run_ledger)
            api.upload_file(
                path_or_fileobj=str(tmp_run),
                path_in_repo=HF_RUN_PATH,
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                commit_message=f"{EXP_ID}: mark RUNNING on Colab {gpu_name}",
            )
        except Exception as exc:
            print(f"[{EXP_ID}] Warning: initial HF ledger upload failed: {exc}", flush=True)

        for idx, piece in enumerate(hw_pieces, start=1):
            slug = piece["slug"]
            ref_page_toks, page_ref_notes = esg.suite31_page_refs(
                piece["gt"], "ttt-" + slug, workdir
            )

            for domain in domains:
                if (slug, domain) in done_keys:
                    continue

                img_path: Path = piece[domain]
                eff_max_dim = max_dim
                with Image.open(img_path) as raw_im:
                    pil_im = uni_omr._resize_fullpage_pil(raw_im.convert("RGB"), eff_max_dim)
                img_t = uni_omr._pil_to_chw_tensor(pil_im).unsqueeze(0).to(device)

                # 1. Exact pre-TTT baseline decode
                ttt.restore()
                before = ttt_mod._decode_and_score_page(
                    model,
                    cfg,
                    img_t,
                    ref_page_toks,
                    page_ref_notes,
                    workdir,
                    f"ttt-before-{slug}-{domain}",
                    max_len=max_len,
                    use_grammar=True,
                    strict_grammar=False,
                    use_mtp_spec=True,
                    use_bf16=use_bf16,
                    device=device,
                )

                row: Dict[str, Any] = {
                    "piece": slug,
                    "handwritten": True,
                    "domain": domain,
                    "eff_max_dim": eff_max_dim,
                    "before": before,
                }

                # 2. Run each requested TTT step count cleanly from restored snapshot
                step_logs: List[str] = []
                for s in step_counts:
                    ttt.restore()
                    try:
                        ttt_stats = ttt.adapt_on_page(img_t, steps=int(s))
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        ttt.restore()
                        # Fallback to 1280px if 1536px autograd hits VRAM ceiling
                        eff_max_dim = 1280
                        with Image.open(img_path) as raw_im:
                            pil_im = uni_omr._resize_fullpage_pil(raw_im.convert("RGB"), eff_max_dim)
                        img_t_fb = uni_omr._pil_to_chw_tensor(pil_im).unsqueeze(0).to(device)
                        ttt_stats = ttt.adapt_on_page(img_t_fb, steps=int(s))
                        ttt_stats["oom_fallback_max_dim"] = eff_max_dim
                        row["eff_max_dim"] = eff_max_dim

                    after = ttt_mod._decode_and_score_page(
                        model,
                        cfg,
                        img_t,
                        ref_page_toks,
                        page_ref_notes,
                        workdir,
                        f"ttt-after-s{s}-{slug}-{domain}",
                        max_len=max_len,
                        use_grammar=True,
                        strict_grammar=False,
                        use_mtp_spec=True,
                        use_bf16=use_bf16,
                        device=device,
                    )
                    ttt.restore()
                    d_note = round(after["note_f1"] - before["note_f1"], 6)
                    d_tok = round(after["tok_f1"] - before["tok_f1"], 6)
                    row[f"after_s{s}"] = after
                    row[f"delta_note_f1_s{s}"] = d_note
                    row[f"delta_tok_f1_s{s}"] = d_tok
                    row[f"ttt_stats_s{s}"] = ttt_stats
                    step_logs.append(
                        f"s{s}:{after['note_f1']:.4f}({d_note:+.4f}, {ttt_stats['elapsed_secs']:.1f}s)"
                    )

                rows.append(row)
                done_keys.add((slug, domain))
                print(
                    f"[{idx:02d}/{len(hw_pieces):02d}] {slug:38s} [{domain:5s}] "
                    f"base={before['note_f1']:.4f} | " + " | ".join(step_logs),
                    flush=True,
                )

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Periodic 5-min checkpoint (and always save after each piece)
                now = time.time()
                if (now - last_save_ts) >= save_interval_s or domain == domains[-1]:
                    _atomic_write_json(partial_path, _build_report("RUNNING"))
                    last_save_ts = now

        # 6. Write final report + upload to Hugging Face Hub
        final_report = _build_report("COMPLETED")
        _atomic_write_json(final_local_path, final_report)
        _atomic_write_json(partial_path, final_report)

        print("=" * 96, flush=True)
        print(f"[{EXP_ID}] FINAL SUMMARY (elapsed={final_report['elapsed_secs']}s)", flush=True)
        print(json.dumps(final_report["summary"], indent=2), flush=True)
        print("=" * 96, flush=True)

        api.upload_file(
            path_or_fileobj=str(final_local_path),
            path_in_repo=HF_EVAL_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"{EXP_ID}: upload final suite31_hw TTT A/B eval JSON (Colab L4)",
        )

        final_ledger = {
            "exp_id": EXP_ID,
            "status": "COMPLETED",
            "started_utc": started_iso,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_secs": final_report["elapsed_secs"],
            "platform": platform_str,
            "hf_eval_path": HF_EVAL_PATH,
            "ckpt_hf_path": HF_CKPT_PATH,
            "summary": final_report["summary"],
        }
        tmp_run = workdir / "run_ledger_final.json"
        _atomic_write_json(tmp_run, final_ledger)
        api.upload_file(
            path_or_fileobj=str(tmp_run),
            path_in_repo=HF_RUN_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"{EXP_ID}: mark COMPLETED with summary (Colab L4)",
        )
        print(
            f"[{EXP_ID}] Uploaded HF artifacts: {HF_EVAL_PATH} & {HF_RUN_PATH}",
            flush=True,
        )
        return final_report["summary"]
    finally:
        if ttt is not None:
            try:
                ttt.restore()
            except Exception:
                pass
        if HF_TOKEN_FILE.exists():
            try:
                HF_TOKEN_FILE.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    run_ttt_experiment()
