#!/usr/bin/env python3
"""Modal L4/A10G/A100 generic experiment runner for `experiments/overnight_queue/queue.json`.

Runs any `ov-*`, `i13-*`, or follow-up `run_colab.py` / `run_exp.py` script unmodified
inside a cached Modal GPU container:
  - Uses `modal.Image.debian_slim(python_version="3.12")` with pinned PyTorch +
    scientific/OMR dependencies and `.add_local_dir("tmp-train/mvd", remote_path="/workspace/mvd")`
    placed LAST for ~2-second code rebuilds.
  - Mounts `modal.Volume.from_name("omr-cache", create_if_missing=True)` at `/vol`
    with `HF_HOME=/vol/hf_cache` so HF datasets & checkpoints cache across runs.
  - Attaches `modal.Secret.from_name("hf-secret")` (provides `HF_TOKEN`; never printed).
  - Wires `/content/mvd`, `/workspace/mvd`, `/workspace/tmp-train/mvd`,
    `/workspace/sheet-music-converter/tmp-train/mvd`, `/content/sheet-music-converter`,
    and `/content/workspace/sheet-music-converter` symlinks so all existing experiment
    scripts run unmodified.
  - Detached launch pattern (`fn.spawn(...)` under `modal run --detach`) so local
    network blips never cancel a running GPU container.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import modal

HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
APP_NAME = "omr-overnight-queue-worker"
DEFAULT_GPU = os.environ.get("MODAL_QUEUE_GPU", "L4")
DEFAULT_TIMEOUT_SECS = int(os.environ.get("MODAL_QUEUE_TIMEOUT", "7200"))

_THIS = Path(__file__).resolve()
REPO_ROOT = _THIS.parents[2] if len(_THIS.parents) > 2 else _THIS.parent

app = modal.App(APP_NAME)
vol = modal.Volume.from_name("omr-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-secret")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "tar", "libglib2.0-0", "libgl1")
    .pip_install(
        "torch==2.14.1",
        "torchvision",
        "huggingface_hub>=0.26",
        "pillow",
        "numpy",
        "opencv-python-headless",
        "scipy",
        "tqdm",
        "music21",
        "verovio",
        "timm",
        "einops",
        "pyyaml",
        "safetensors",
    )
    .env(
        {
            "HF_HOME": "/vol/hf_cache",
            "HUGGINGFACE_HUB_CACHE": "/vol/hf_cache/hub",
            "PYTHONUNBUFFERED": "1",
        }
    )
    # Keep local repo code LAST so edits rebuild in ~2s instead of re-installing PyTorch.
    .add_local_dir(
        "tmp-train/mvd",
        remote_path="/workspace/mvd",
        ignore=[
            "__pycache__",
            "**/__pycache__",
            "*.pyc",
            "*.pyo",
            "*.pt",
            "*.onnx",
            "*.png",
            "*.jpg",
            "*.npz",
            "*.npy",
            "*.tar",
        ],
    )
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ensure_symlink(link_path: Path, target: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink() or link_path.exists():
        try:
            if link_path.is_symlink() and link_path.resolve() == target.resolve():
                return
        except OSError:
            pass
        if link_path.is_symlink() or link_path.is_file():
            link_path.unlink()
        else:
            shutil.rmtree(link_path)
    link_path.symlink_to(target)


def _hf_eval_exists(hf_eval_target: str, token: Optional[str]) -> bool:
    if not hf_eval_target:
        return False
    try:
        from huggingface_hub import HfApi  # type: ignore

        api = HfApi(token=token) if token else HfApi()
        return bool(api.file_exists(HF_REPO_ID, hf_eval_target, repo_type="dataset"))
    except Exception as exc:
        print(f"[modal-runner] WARN: HF existence check failed for {hf_eval_target}: {exc}", flush=True)
        return False


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    cpu=4,
    memory=24576,
    timeout=DEFAULT_TIMEOUT_SECS,
    volumes={"/vol": vol},
    secrets=[hf_secret],
)
def run_experiment(
    exp_id: str,
    script_source: str,
    script_basename: str = "run_colab.py",
    hf_eval_target: str = "",
) -> Dict[str, Any]:
    """Execute one experiment script inside a Modal GPU container."""
    t0 = time.monotonic()
    started_at = _utc_now()

    vol.reload()

    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        raise RuntimeError("HF_TOKEN missing inside Modal container (check hf-secret)")
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

    # Ensure persistent HF cache dirs exist on /vol
    hf_cache_root = Path("/vol/hf_cache")
    (hf_cache_root / "hub").mkdir(parents=True, exist_ok=True)
    vol_logs_dir = Path("/vol/logs")
    vol_logs_dir.mkdir(parents=True, exist_ok=True)

    # Probe GPU (never print secrets)
    gpu_info = "unknown"
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            cc = torch.cuda.get_device_capability(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_info = f"{name} (cc={cc[0]}.{cc[1]}, {vram_gb:.1f} GiB, native_bf16={cc[0] >= 8})"
        else:
            gpu_info = "NO_CUDA"
    except Exception as exc:
        gpu_info = f"probe_error:{exc}"

    print(
        f"[modal:{exp_id}] started={started_at} gpu={gpu_info} HF_HOME={os.environ.get('HF_HOME')}",
        flush=True,
    )

    # Fast-skip if eval artifact already landed on HF
    if hf_eval_target and _hf_eval_exists(hf_eval_target, hf_token):
        elapsed = round(time.monotonic() - t0, 2)
        print(f"[modal:{exp_id}] Eval {hf_eval_target} already exists on HF; skipping.", flush=True)
        return {
            "exp_id": exp_id,
            "returncode": 0,
            "skipped_existing_hf": True,
            "hf_eval_exists": True,
            "gpu_info": gpu_info,
            "started_at": started_at,
            "completed_at": _utc_now(),
            "elapsed_sec": elapsed,
            "log_tail": f"[modal:{exp_id}] skipped (already exists on HF: {hf_eval_target})\n",
        }

    # Wire every path convention used across ov-* and i13-* scripts
    mvd_src = Path("/workspace/mvd")
    if not mvd_src.is_dir():
        raise FileNotFoundError(f"Missing mounted mvd directory: {mvd_src}")

    repo_root = Path("/workspace/sheet-music-converter")
    repo_root.mkdir(parents=True, exist_ok=True)

    # Per-container scratch dir for /content/vol so parallel containers never race on tar extraction
    container_vol_scratch = Path("/tmp/modal_content_vol") / exp_id
    container_vol_scratch.mkdir(parents=True, exist_ok=True)
    Path("/content").mkdir(parents=True, exist_ok=True)

    _ensure_symlink(Path("/content/mvd"), mvd_src)
    _ensure_symlink(Path("/workspace/tmp-train/mvd"), mvd_src)
    _ensure_symlink(repo_root / "tmp-train" / "mvd", mvd_src)
    _ensure_symlink(Path("/content/sheet-music-converter"), repo_root)
    _ensure_symlink(Path("/content/workspace/sheet-music-converter"), repo_root)
    _ensure_symlink(Path("/content/vol"), container_vol_scratch)
    _ensure_symlink(Path("/content/cache"), hf_cache_root)

    # Materialize experiment script at repo_root/experiments/<exp_id>/<script_basename>
    # so Path(__file__).resolve().parents[2] resolves cleanly to repo_root.
    exec_script = repo_root / "experiments" / exp_id / script_basename
    exec_script.parent.mkdir(parents=True, exist_ok=True)
    exec_script.write_text(script_source, encoding="utf-8", newline="\n")

    env = os.environ.copy()
    py_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{mvd_src}:{py_path}" if py_path else str(mvd_src)
    env["PYTHONUNBUFFERED"] = "1"
    env["HF_HOME"] = str(hf_cache_root)
    env["HUGGINGFACE_HUB_CACHE"] = str(hf_cache_root / "hub")

    local_log_path = Path("/tmp") / f"{exp_id}.modal.log"
    print(f"[modal:{exp_id}] Launching {exec_script} (cwd={repo_root})", flush=True)

    rc = -1
    max_return_chars = 2_000_000  # 2 MB cap on RPC return payload
    captured_chunks: list[str] = []
    captured_len = 0

    try:
        with open(local_log_path, "w", encoding="utf-8", newline="\n") as log_f:
            header = (
                f"=== MODAL {exp_id} | gpu={gpu_info} | started={started_at} ===\n"
            )
            log_f.write(header)
            captured_chunks.append(header)
            captured_len += len(header)

            proc = subprocess.Popen(
                [sys.executable, "-u", str(exec_script)],
                cwd=str(repo_root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="", flush=True)
                log_f.write(line)
                if captured_len < max_return_chars:
                    captured_chunks.append(line)
                    captured_len += len(line)
            rc = int(proc.wait())
            footer = f"\n[modal:{exp_id}] Finished rc={rc} elapsed={time.monotonic() - t0:.1f}s\n"
            print(footer, end="", flush=True)
            log_f.write(footer)
            captured_chunks.append(footer)
    finally:
        # Persist log + any newly downloaded HF cache blobs to omr-cache volume
        try:
            if local_log_path.is_file():
                shutil.copyfile(local_log_path, vol_logs_dir / f"{exp_id}.modal.log")
            vol.commit()
        except Exception as vol_exc:
            print(f"[modal:{exp_id}] WARN: vol.commit() failed: {vol_exc}", flush=True)

    hf_ok = _hf_eval_exists(hf_eval_target, hf_token) if hf_eval_target else (rc == 0)
    elapsed = round(time.monotonic() - t0, 2)
    return {
        "exp_id": exp_id,
        "returncode": rc,
        "skipped_existing_hf": False,
        "hf_eval_exists": hf_ok,
        "gpu_info": gpu_info,
        "started_at": started_at,
        "completed_at": _utc_now(),
        "elapsed_sec": elapsed,
        "log_tail": "".join(captured_chunks),
    }


@app.local_entrypoint()
def main(
    exp_id: str,
    script_path: str,
    hf_eval_target: str = "",
    gpu: str = DEFAULT_GPU,
    timeout_secs: int = DEFAULT_TIMEOUT_SECS,
) -> None:
    """Spawn one experiment detached and emit machine-parseable call/app IDs."""
    sp = Path(script_path)
    if not sp.is_absolute():
        sp = REPO_ROOT / sp
    if not sp.is_file():
        raise FileNotFoundError(f"Experiment script not found: {sp}")

    script_source = sp.read_text(encoding="utf-8")
    fn = run_experiment.with_options(gpu=gpu, timeout=int(timeout_secs))
    call = fn.spawn(
        exp_id=exp_id,
        script_source=script_source,
        script_basename=sp.name,
        hf_eval_target=hf_eval_target,
    )
    payload = {
        "exp_id": exp_id,
        "app_id": app.app_id,
        "call_id": call.object_id,
        "gpu": gpu,
        "timeout_secs": int(timeout_secs),
    }
    print(f"MODAL_SPAWN_JSON={json.dumps(payload)}", flush=True)
