#!/usr/bin/env python3
"""Google Colab L4 runner for Overnight Exp 33: V-JEPA 2.1 Warm-Start vs Scratch Transfer Probe.

Experiment ID: 2026-11-26T01-38-ov-33-vjepa-warmstart-vs-scratch
Compares two matched 800-step UniOMR v2 small_45m runs on datasets/iter10_5_proxy_2k.tar:
  - arm_a_scratch: Random initialization
  - arm_b_vjepa_warmstart: --init-encoder-from RISM V-JEPA 2.1 checkpoint
    (fetched from HF Exp 2 if present, else generated via a 400-step RISM warmup)

Evaluates both arms on datasets/suite31_hw_eval.tar (45 pieces: 31 printed + 14 real HW,
clean + photo = 90 pages) and uploads results to Hugging Face Private Dataset Hub:
  - evals/2026-11-26T01-38-ov-33-vjepa-warmstart-vs-scratch.json
  - runs/2026-11-26T01-38-ov-33-vjepa-warmstart-vs-scratch.json
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

EXP_ID = "2026-11-26T01-38-ov-33-vjepa-warmstart-vs-scratch"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_PROXY_TAR = "datasets/iter10_5_proxy_2k.tar"
HF_RISM_TAR = "datasets/rism_hw_5707.tar"
HF_SUITE31_TAR = "datasets/suite31_hw_eval.tar"
HF_EXP2_CKPT_CANDIDATES = (
    "checkpoints/2026-11-26T00-54-vjepa21-gram-rism-v2/uni_omr_v2_ijepa_best.pt",
    "checkpoints/2026-11-26T00-54-vjepa21-gram-rism-v2/uni_omr_v2_ijepa_last.pt",
)
HF_EVAL_PATH = f"evals/{EXP_ID}.json"
HF_RUN_PATH = f"runs/{EXP_ID}.json"

HF_TOKEN_FILE = Path("/tmp/.hf_token")
WORKSPACE_ROOT = Path("/workspace")
VOL_ROOT = Path("/content/vol")

# Controlled transfer hyperparameters
TRAIN_STEPS = 800
WARMUP_SSL_STEPS = 400
BATCH_SIZE = 2
GRAD_ACCUM = 4
FULLPAGE_MAX_DIM = 1280
MAX_SEQ_LEN = 1536
LR = 3e-4
WARMUP_STEPS = 80
EVAL_EVERY = 400
SAVE_EVERY_MINUTES = 5.0
SUITE31_MAX_DIM = 1280
SUITE31_MAX_LEN = 768
SEED = 42


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _run_cmd(
    cmd: Sequence[str] | str,
    *,
    shell: bool = False,
    env: Optional[Dict[str, str]] = None,
) -> None:
    label = cmd if isinstance(cmd, str) else " ".join(cmd)
    print(f"[{EXP_ID}] $ {label}", flush=True)
    proc = subprocess.Popen(
        cmd if shell else list(cmd),
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
        raise RuntimeError(f"Command failed (rc={rc}): {label}")


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


def _locate_or_extract_repo_root() -> Path:
    candidates = [
        WORKSPACE_ROOT / "sheet-music-converter",
        Path("/content/sheet-music-converter"),
        Path.cwd(),
    ]
    for cand in candidates:
        if (cand / "tmp-train" / "mvd" / "train_uni_v2.py").is_file():
            return cand

    for bundle in (Path("/content/repo_bundle.tar.gz"), Path("/content/bundle.tar.gz")):
        if bundle.is_file():
            WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
            print(f"[{EXP_ID}] Extracting {bundle} -> {WORKSPACE_ROOT}...", flush=True)
            with tarfile.open(bundle, "r:*") as tf:
                tf.extractall(path=WORKSPACE_ROOT)
            for cand in candidates:
                if (cand / "tmp-train" / "mvd" / "train_uni_v2.py").is_file():
                    return cand

    raise FileNotFoundError(
        "Could not locate sheet-music-converter/tmp-train/mvd/train_uni_v2.py. "
        "Upload repo tarball to /content/repo_bundle.tar.gz before executing."
    )


def _ensure_packages() -> None:
    needed = []
    for mod, pkg in [
        ("huggingface_hub", "huggingface_hub>=0.26"),
        ("PIL", "pillow"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("tqdm", "tqdm"),
    ]:
        try:
            __import__(mod)
        except ImportError:
            needed.append(pkg)
    if needed:
        _run_cmd([sys.executable, "-m", "pip", "install", "-q", *needed])


def _maybe_build_rust_scorers(repo_root: Path) -> Tuple[Optional[Path], Optional[Path]]:
    aser_manifest = repo_root / "converter" / "assembly-serialize" / "Cargo.toml"
    eharness_manifest = repo_root / "converter" / "eval-harness" / "Cargo.toml"
    aser_bin = aser_manifest.parent / "target" / "release" / "assembly-serialize"
    eharness_bin = eharness_manifest.parent / "target" / "release" / "eval-harness"
    if not (aser_manifest.is_file() and eharness_manifest.is_file()):
        return None, None
    try:
        cargo_bin_dir = Path.home() / ".cargo" / "bin"
        env = os.environ.copy()
        env["PATH"] = f"{cargo_bin_dir}:{env.get('PATH', '')}"
        if shutil.which("cargo", path=env["PATH"]) is None:
            print(f"[{EXP_ID}] Installing minimal Rust toolchain for Strict Note F1...", flush=True)
            _run_cmd(
                "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
                "| sh -s -- -y --default-toolchain stable --profile minimal",
                shell=True,
                env=env,
            )
        if not aser_bin.is_file():
            _run_cmd(["cargo", "build", "--release", "--manifest-path", str(aser_manifest)], env=env)
        if not eharness_bin.is_file():
            _run_cmd(["cargo", "build", "--release", "--manifest-path", str(eharness_manifest)], env=env)
        if aser_bin.is_file() and eharness_bin.is_file():
            return aser_bin, eharness_bin
    except Exception as exc:
        print(f"[{EXP_ID}] Rust scorer build skipped ({exc}); continuing with Token F1.", flush=True)
    return None, None


def _ensure_proxy_and_suite31(hf_token: str) -> Tuple[Path, Path, Path]:
    from huggingface_hub import hf_hub_download

    ds_root = VOL_ROOT / "datasets"
    ds_root.mkdir(parents=True, exist_ok=True)

    # 1. iter10_5_proxy_2k
    proxy_dir = ds_root / "iter10_5_proxy_2k"
    train_abs = proxy_dir / "proxy_train_abs.jsonl"
    val_abs = proxy_dir / "proxy_val_abs.jsonl"
    if not (train_abs.is_file() and val_abs.is_file()):
        proxy_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{EXP_ID}] Downloading {HF_PROXY_TAR}...", flush=True)
        tar_p = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_PROXY_TAR,
            repo_type="dataset",
            token=hf_token,
            local_dir="/tmp/hf_dl",
        )
        with tarfile.open(tar_p, "r:*") as tf:
            tf.extractall(path=proxy_dir)
        try:
            os.remove(tar_p)
        except OSError:
            pass
        for src_name, dst_path in (("proxy_train.jsonl", train_abs), ("proxy_val.jsonl", val_abs)):
            src_path = proxy_dir / src_name
            out_lines: List[str] = []
            for raw in src_path.read_text(encoding="utf-8").splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                rec = json.loads(raw)
                rec["image"] = str((proxy_dir / rec["image"]).resolve())
                rec["bekerns"] = [str((proxy_dir / b).resolve()) for b in rec.get("bekerns", [])]
                out_lines.append(json.dumps(rec))
            dst_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    # 2. suite31_hw_eval
    suite31_dir = ds_root / "suite31_hw_eval"
    manifest_path = suite31_dir / "suite31" / "suite31_hw_manifest.json"
    if not manifest_path.is_file():
        suite31_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{EXP_ID}] Downloading {HF_SUITE31_TAR}...", flush=True)
        tar_p = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_SUITE31_TAR,
            repo_type="dataset",
            token=hf_token,
            local_dir="/tmp/hf_dl",
        )
        with tarfile.open(tar_p, "r:*") as tf:
            tf.extractall(path=suite31_dir)
        try:
            os.remove(tar_p)
        except OSError:
            pass

    return train_abs, val_abs, manifest_path


def _resolve_or_pretrain_vjepa_ckpt(
    hf_token: str,
    mvd_dir: Path,
) -> Tuple[Path, str]:
    """Fetch Exp 2 V-JEPA 2.1 checkpoint from HF if present, else run a 400-step RISM warmup."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi(token=hf_token)
    repo_files = set(api.list_repo_files(HF_REPO_ID, repo_type="dataset"))
    for cand in HF_EXP2_CKPT_CANDIDATES:
        if cand in repo_files:
            print(f"[{EXP_ID}] Found existing V-JEPA 2.1 HF checkpoint: {cand}", flush=True)
            local_p = Path(
                hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename=cand,
                    repo_type="dataset",
                    token=hf_token,
                    local_dir=str(VOL_ROOT / "ckpts" / "ov33" / "hf_init"),
                )
            )
            return local_p, f"hf:{cand}"

    print(
        f"[{EXP_ID}] Exp 2 HF checkpoint not found; running {WARMUP_SSL_STEPS}-step RISM V-JEPA 2.1 warmup...",
        flush=True,
    )
    rism_dir = VOL_ROOT / "datasets" / "rism_hw_5707"
    sentinel = rism_dir / ".unpacked_ok"
    if not sentinel.is_file():
        rism_dir.mkdir(parents=True, exist_ok=True)
        tar_p = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_RISM_TAR,
            repo_type="dataset",
            token=hf_token,
            local_dir="/tmp/hf_dl",
        )
        with tarfile.open(tar_p, "r:*") as tf:
            tf.extractall(path=rism_dir)
        try:
            os.remove(tar_p)
        except OSError:
            pass
        sentinel.write_text("ok\n", encoding="utf-8")

    warmup_out = VOL_ROOT / "ckpts" / "ov33" / "rism_warmup"
    warmup_out.mkdir(parents=True, exist_ok=True)
    _run_cmd(
        [
            sys.executable,
            "-u",
            str(mvd_dir / "pretrain_ijepa_rism_v2.py"),
            "--preset",
            "small_45m",
            "--image-dir",
            str(rism_dir),
            "--out-dir",
            str(warmup_out),
            "--steps",
            str(WARMUP_SSL_STEPS),
            "--batch-size",
            "4",
            "--grad-accum",
            "2",
            "--max-dim",
            "1024",
            "--bf16",
            "--optimizer",
            "muon",
            "--gram-weight",
            "0.5",
            "--stroke-hog-weight",
            "0.25",
            "--save-every-minutes",
            str(SAVE_EVERY_MINUTES),
            "--log-every",
            "25",
            "--seed",
            str(SEED),
        ]
    )
    ckpt = warmup_out / "uni_omr_v2_ijepa_last.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"Expected RISM warmup checkpoint at {ckpt}")
    return ckpt, f"rism_warmup_{WARMUP_SSL_STEPS}steps"


def _extract_suite31_summary(eval_json_path: Path) -> Dict[str, Any]:
    raw = json.loads(eval_json_path.read_text(encoding="utf-8"))
    summary = raw.get("summary") or raw.get("splits") or raw
    out: Dict[str, Any] = {}
    for split_key in ("hw_clean", "hw_photo", "printed_clean", "printed_photo", "all45_clean", "all45_photo"):
        if isinstance(summary, dict) and split_key in summary and isinstance(summary[split_key], dict):
            s = summary[split_key]
            out[split_key] = {
                "note_f1": s.get("note_f1", s.get("strict_note_f1")),
                "tok_f1": s.get("tok_f1", s.get("token_f1")),
                "ser": s.get("ser"),
                "n": s.get("n", s.get("count")),
            }
    if not out:
        out["raw_keys"] = sorted(list(raw.keys()))[:20]
    return out


def _train_and_eval_arm(
    arm_name: str,
    mvd_dir: Path,
    train_manifest: Path,
    suite31_manifest: Path,
    init_encoder_from: Optional[Path],
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    arm_dir = VOL_ROOT / "ckpts" / "ov33" / arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)

    train_cmd = [
        sys.executable,
        "-u",
        str(mvd_dir / "train_uni_v2.py"),
        "--preset",
        "small_45m",
        "--curriculum",
        "fullpage",
        "--fullpage-manifest",
        str(train_manifest),
        "--out-dir",
        str(arm_dir),
        "--steps",
        str(TRAIN_STEPS),
        "--batch-size",
        str(BATCH_SIZE),
        "--grad-accum",
        str(GRAD_ACCUM),
        "--fullpage-max-dim",
        str(FULLPAGE_MAX_DIM),
        "--max-seq-len",
        str(MAX_SEQ_LEN),
        "--optimizer",
        "muon",
        "--lr",
        str(LR),
        "--lr-schedule",
        "wsd",
        "--warmup-steps",
        str(WARMUP_STEPS),
        "--bf16",
        "--grad-checkpoint",
        "--eval-every",
        str(EVAL_EVERY),
        "--save-every-minutes",
        str(SAVE_EVERY_MINUTES),
        "--log-every",
        "25",
        "--seed",
        str(SEED),
    ]
    if init_encoder_from is not None:
        train_cmd.extend(["--init-encoder-from", str(init_encoder_from)])

    print(f"\n[{EXP_ID}] === Training {arm_name} (init_encoder={init_encoder_from}) ===", flush=True)
    _run_cmd(train_cmd)
    train_secs = round(time.perf_counter() - t0, 2)

    # Prefer EMA best/latest, else raw latest
    ckpt_path = None
    for name in ("uni_omr_v2_ema_best.pt", "uni_omr_v2_ema_latest.pt", "uni_omr_v2_latest.pt"):
        cand = arm_dir / name
        if cand.is_file():
            ckpt_path = cand
            break
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint produced in {arm_dir}")

    train_log_path = arm_dir / "train_log.json"
    train_tail: List[Dict[str, Any]] = []
    if train_log_path.is_file():
        hist = json.loads(train_log_path.read_text(encoding="utf-8"))
        if isinstance(hist, list):
            train_tail = hist[-5:]

    suite31_out = arm_dir / f"{arm_name}_suite31_hw.json"
    print(f"[{EXP_ID}] === Evaluating {arm_name} on suite31_hw ({ckpt_path.name}) ===", flush=True)
    t_eval = time.perf_counter()
    _run_cmd(
        [
            sys.executable,
            "-u",
            str(mvd_dir / "eval_uni_suite31.py"),
            "--ckpt",
            str(ckpt_path),
            "--manifest",
            str(suite31_manifest),
            "--max-dim",
            str(SUITE31_MAX_DIM),
            "--max-len",
            str(SUITE31_MAX_LEN),
            "--bf16",
            "--out",
            str(suite31_out),
        ]
    )
    eval_secs = round(time.perf_counter() - t_eval, 2)
    suite31_summary = _extract_suite31_summary(suite31_out)

    return {
        "arm": arm_name,
        "init_encoder_from": str(init_encoder_from) if init_encoder_from else None,
        "evaluated_ckpt": ckpt_path.name,
        "train_secs": train_secs,
        "eval_secs": eval_secs,
        "train_log_tail": train_tail,
        "suite31_hw": suite31_summary,
    }


def _delta(b: Optional[float], a: Optional[float]) -> Optional[float]:
    if b is None or a is None:
        return None
    return round(float(b) - float(a), 6)


def main() -> None:
    t_start = time.time()
    started_iso = datetime.now(timezone.utc).isoformat()
    try:
        hf_token = _read_and_purge_hf_token()
        _ensure_packages()
        repo_root = _locate_or_extract_repo_root()
        mvd_dir = repo_root / "tmp-train" / "mvd"

        import torch
        from huggingface_hub import HfApi

        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        print(f"[{EXP_ID}] GPU={gpu_name}, bf16={use_bf16}", flush=True)

        aser_bin, eharness_bin = _maybe_build_rust_scorers(repo_root)
        has_rust = bool(aser_bin and eharness_bin)
        print(f"[{EXP_ID}] Rust Strict Note F1 scorers available={has_rust}", flush=True)

        train_abs, _val_abs, suite31_manifest = _ensure_proxy_and_suite31(hf_token)
        vjepa_ckpt, vjepa_source = _resolve_or_pretrain_vjepa_ckpt(hf_token, mvd_dir)

        arm_a = _train_and_eval_arm(
            "arm_a_scratch",
            mvd_dir=mvd_dir,
            train_manifest=train_abs,
            suite31_manifest=suite31_manifest,
            init_encoder_from=None,
        )
        arm_b = _train_and_eval_arm(
            "arm_b_vjepa_warmstart",
            mvd_dir=mvd_dir,
            train_manifest=train_abs,
            suite31_manifest=suite31_manifest,
            init_encoder_from=vjepa_ckpt,
        )

        comparison: Dict[str, Any] = {"vjepa_source": vjepa_source}
        for split_key in ("hw_clean", "hw_photo", "printed_clean", "printed_photo", "all45_clean", "all45_photo"):
            sa = arm_a["suite31_hw"].get(split_key, {})
            sb = arm_b["suite31_hw"].get(split_key, {})
            if isinstance(sa, dict) and isinstance(sb, dict):
                comparison[f"{split_key}_note_f1_delta"] = _delta(sb.get("note_f1"), sa.get("note_f1"))
                comparison[f"{split_key}_tok_f1_delta"] = _delta(sb.get("tok_f1"), sa.get("tok_f1"))

        report = {
            "exp_id": EXP_ID,
            "status": "completed",
            "started_utc": started_iso,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_secs": round(time.time() - t_start, 2),
            "platform": f"colab-{gpu_name}",
            "bf16": use_bf16,
            "has_rust_scorer": has_rust,
            "vjepa_source": vjepa_source,
            "hyperparams": {
                "preset": "small_45m",
                "train_steps": TRAIN_STEPS,
                "warmup_ssl_steps_if_fallback": WARMUP_SSL_STEPS,
                "batch_size": BATCH_SIZE,
                "grad_accum": GRAD_ACCUM,
                "fullpage_max_dim": FULLPAGE_MAX_DIM,
                "max_seq_len": MAX_SEQ_LEN,
                "lr": LR,
                "seed": SEED,
            },
            "arm_a_scratch": arm_a,
            "arm_b_vjepa_warmstart": arm_b,
            "comparison_warmstart_minus_scratch": comparison,
        }

        eval_local = VOL_ROOT / "evals" / f"{EXP_ID}.json"
        run_local = VOL_ROOT / "runs" / f"{EXP_ID}.json"
        _atomic_write_json(eval_local, report)
        _atomic_write_json(
            run_local,
            {
                "exp_id": EXP_ID,
                "status": "completed",
                "platform": f"colab-{gpu_name}",
                "started_utc": started_iso,
                "completed_utc": report["completed_utc"],
                "elapsed_secs": report["elapsed_secs"],
                "eval_path": HF_EVAL_PATH,
                "vjepa_source": vjepa_source,
                "comparison_warmstart_minus_scratch": comparison,
            },
        )

        api = HfApi(token=hf_token)
        api.upload_file(
            path_or_fileobj=str(eval_local),
            path_in_repo=HF_EVAL_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"eval({EXP_ID}): upload V-JEPA warmstart vs scratch transfer probe",
        )
        api.upload_file(
            path_or_fileobj=str(run_local),
            path_in_repo=HF_RUN_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"run({EXP_ID}): upload V-JEPA warmstart vs scratch run ledger",
        )
        print(f"\n[{EXP_ID}] UPLOADED {HF_EVAL_PATH} & {HF_RUN_PATH} TO HF!", flush=True)
        print(json.dumps(comparison, indent=2), flush=True)
    finally:
        if HF_TOKEN_FILE.exists():
            try:
                HF_TOKEN_FILE.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    main()
