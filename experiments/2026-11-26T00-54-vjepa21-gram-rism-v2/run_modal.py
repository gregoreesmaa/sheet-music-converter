"""Modal L4 launcher for Experiment 2: V-JEPA 2.1 + DINOv3 Gram Anchoring + Stroke-HOG on RISM HW (5,707 pages).

Usage (detached, recommended):
    .\\.venv\\Scripts\\modal.exe run --detach experiments/2026-11-26T00-54-vjepa21-gram-rism-v2/run_modal.py

Resume from Modal Volume autosave checkpoint:
    .\\.venv\\Scripts\\modal.exe run --detach experiments/2026-11-26T00-54-vjepa21-gram-rism-v2/run_modal.py \\
        --init-from /vol/ckpts/exp2_vjepa21_rism_v2/uni_omr_v2_ijepa_autosave.pt
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from typing import Any, Dict

import modal

EXP_ID = "2026-11-26T00-54-vjepa21-gram-rism-v2"
APP_NAME = "exp2-vjepa21-gram-rism-v2"
HF_REPO = "gregoreesmaa/sheet-music-converter-omr"
HF_TAR_PATH = "datasets/rism_hw_5707.tar"
VOL_DATASET_DIR = "/vol/datasets/rism_hw"
VOL_DATASET_MARKER = "/vol/datasets/rism_hw/.unpacked_rism_hw_5707"
VOL_CKPT_DIR = "/vol/ckpts/exp2_vjepa21_rism_v2"

app = modal.App(APP_NAME)
vol = modal.Volume.from_name("omr-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-secret")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "torchvision",
        "numpy",
        "pillow",
        "opencv-python-headless",
        "huggingface_hub",
    )
    .add_local_dir("tmp-train/mvd", remote_path="/app/mvd")
)


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _upload_file_safe(api: Any, local_path: Path, path_in_repo: str, commit_message: str) -> bool:
    if not local_path.is_file():
        print(f"[HF skip] missing {local_path}", flush=True)
        return False
    try:
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_id=HF_REPO,
            repo_type="dataset",
            commit_message=commit_message,
        )
        size_mb = local_path.stat().st_size / (1024.0 * 1024.0)
        print(f"[HF upload] {path_in_repo} ({size_mb:.2f} MB)", flush=True)
        return True
    except Exception as exc:
        print(f"[HF warn] failed uploading {path_in_repo}: {exc}", flush=True)
        return False


def _upload_run_ledger(api: Any, payload: Dict[str, Any], commit_message: str) -> None:
    tmp_json = Path("/tmp") / f"{EXP_ID}_ledger.json"
    tmp_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _upload_file_safe(api, tmp_json, f"runs/{EXP_ID}.json", commit_message)


def _ensure_rism_dataset(hf_token: str) -> int:
    from huggingface_hub import hf_hub_download

    marker = Path(VOL_DATASET_MARKER)
    ds_dir = Path(VOL_DATASET_DIR)
    if marker.is_file():
        print(f"[data] RISM HW already cached at {ds_dir} (marker={marker.read_text().strip()})", flush=True)
        return int(marker.read_text().strip().split()[0])

    print(f"[data] Downloading {HF_TAR_PATH} from {HF_REPO}...", flush=True)
    dl_dir = Path("/tmp/hf_rism_dl")
    dl_dir.mkdir(parents=True, exist_ok=True)
    tar_local = Path(
        hf_hub_download(
            repo_id=HF_REPO,
            filename=HF_TAR_PATH,
            repo_type="dataset",
            token=hf_token,
            local_dir=str(dl_dir),
        )
    )
    ds_dir.mkdir(parents=True, exist_ok=True)
    print(f"[data] Extracting {tar_local} -> {ds_dir}...", flush=True)
    t0 = time.perf_counter()
    with tarfile.open(tar_local, "r:*") as tf:
        tf.extractall(path=ds_dir)
    shutil.rmtree(dl_dir, ignore_errors=True)

    exts = {".png", ".jpg", ".jpeg", ".webp"}
    n_imgs = sum(
        1
        for _root, _dirs, files in os.walk(ds_dir)
        for fn in files
        if os.path.splitext(fn)[1].lower() in exts
    )
    tmp_marker = Path(VOL_DATASET_MARKER + ".tmp")
    tmp_marker.write_text(f"{n_imgs} images unpacked at {_utc_now_iso()}\n", encoding="utf-8")
    os.replace(tmp_marker, marker)
    vol.commit()
    print(
        f"[data] Unpacked {n_imgs} images in {time.perf_counter() - t0:.1f}s and committed to Modal Volume.",
        flush=True,
    )
    return n_imgs


@app.function(
    image=image,
    gpu="L4",
    timeout=7200,
    volumes={"/vol": vol},
    secrets=[hf_secret],
)
def run_vjepa21_pretrain(steps: int = 1500, init_from: str = "") -> Dict[str, Any]:
    import torch
    from huggingface_hub import HfApi

    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        raise RuntimeError("HF_TOKEN secret missing in container environment.")
    api = HfApi(token=hf_token)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    cc = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
    vram_gb = (
        torch.cuda.get_device_properties(0).total_memory / (1024.0 ** 3)
        if torch.cuda.is_available()
        else 0.0
    )
    print(
        f"[env] exp={EXP_ID} torch={torch.__version__} muon={hasattr(torch.optim, 'Muon')} "
        f"gpu={gpu_name} cc={cc[0]}.{cc[1]} vram={vram_gb:.2f}GB",
        flush=True,
    )

    n_imgs = _ensure_rism_dataset(hf_token)
    out_dir = Path(VOL_CKPT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    started_at = _utc_now_iso()
    ledger: Dict[str, Any] = {
        "exp_id": EXP_ID,
        "app_name": APP_NAME,
        "provider": "modal",
        "gpu": gpu_name,
        "cc": f"{cc[0]}.{cc[1]}",
        "torch_version": torch.__version__,
        "num_images": n_imgs,
        "steps": int(steps),
        "init_from": init_from or None,
        "status": "running",
        "started_at": started_at,
        "updated_at": started_at,
    }
    _upload_run_ledger(api, ledger, f"ledger({EXP_ID}): mark running on Modal {gpu_name}")

    cmd = [
        sys.executable,
        "-u",
        "/app/mvd/pretrain_ijepa_rism_v2.py",
        "--preset",
        "small_45m",
        "--image-dir",
        VOL_DATASET_DIR,
        "--out-dir",
        VOL_CKPT_DIR,
        "--steps",
        str(int(steps)),
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
        "1.0",
        "--stroke-hog-weight",
        "0.25",
        "--save-every-minutes",
        "5.0",
        "--save-every",
        "500",
        "--log-every",
        "25",
    ]
    if init_from:
        cmd.extend(["--init-from", init_from])

    stop_sync = threading.Event()

    def _periodic_sync_loop() -> None:
        last_hf_mtime = 0.0
        last_hf_push = time.monotonic()
        while not stop_sync.wait(300.0):
            try:
                vol.commit()
                print(f"[sync] Modal Volume committed at {_utc_now_iso()}", flush=True)
            except Exception as exc:
                print(f"[sync warn] vol.commit failed: {exc}", flush=True)

            autosave_pt = out_dir / "uni_omr_v2_ijepa_autosave.pt"
            now_m = time.monotonic()
            if autosave_pt.is_file() and (now_m - last_hf_push) >= 1200.0:
                mtime = autosave_pt.stat().st_mtime
                if mtime > last_hf_mtime:
                    _upload_file_safe(
                        api,
                        autosave_pt,
                        f"checkpoints/{EXP_ID}/uni_omr_v2_ijepa_autosave.pt",
                        f"ckpt({EXP_ID}): rolling autosave sync",
                    )
                    last_hf_mtime = mtime
                    last_hf_push = now_m

    sync_thread = threading.Thread(target=_periodic_sync_loop, daemon=True)
    sync_thread.start()

    t_start = time.perf_counter()
    print(f"[launch] {' '.join(cmd)}", flush=True)
    rc = 1
    try:
        proc = subprocess.run(cmd, check=False)
        rc = int(proc.returncode)
    finally:
        stop_sync.set()
        sync_thread.join(timeout=15.0)
        vol.commit()

    elapsed_s = time.perf_counter() - t_start
    log_path = out_dir / "vjepa21_train_log.json"
    summary: Dict[str, Any] = {}
    if log_path.is_file():
        try:
            summary = json.loads(log_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[warn] could not parse {log_path}: {exc}", flush=True)

    # Upload final artifacts to Hugging Face
    uploaded: Dict[str, bool] = {}
    uploaded["last_pt"] = _upload_file_safe(
        api,
        out_dir / "uni_omr_v2_ijepa_last.pt",
        f"checkpoints/{EXP_ID}/uni_omr_v2_ijepa_last.pt",
        f"ckpt({EXP_ID}): final V-JEPA 2.1 + Gram + Stroke-HOG encoder",
    )
    if (out_dir / "uni_omr_v2_ijepa_best.pt").is_file():
        uploaded["best_pt"] = _upload_file_safe(
            api,
            out_dir / "uni_omr_v2_ijepa_best.pt",
            f"checkpoints/{EXP_ID}/uni_omr_v2_ijepa_best.pt",
            f"ckpt({EXP_ID}): best EMA loss V-JEPA 2.1 encoder",
        )
    uploaded["train_log"] = _upload_file_safe(
        api,
        log_path,
        f"checkpoints/{EXP_ID}/vjepa21_train_log.json",
        f"log({EXP_ID}): training log",
    )

    ledger.update(
        {
            "status": "completed" if rc == 0 else "failed",
            "returncode": rc,
            "elapsed_sec": round(elapsed_s, 1),
            "updated_at": _utc_now_iso(),
            "best_ema_loss": summary.get("best_ema_loss"),
            "final_ema_loss": summary.get("final_ema_loss"),
            "optimizer": summary.get("optimizer"),
            "oom_skips": summary.get("oom_skips"),
            "uploaded": uploaded,
        }
    )
    _upload_run_ledger(api, ledger, f"ledger({EXP_ID}): mark {ledger['status']} (rc={rc})")

    if rc != 0:
        raise RuntimeError(f"pretrain_ijepa_rism_v2.py exited with code {rc}")
    return ledger


@app.local_entrypoint()
def main(steps: int = 1500, init_from: str = "") -> None:
    result = run_vjepa21_pretrain.remote(steps=steps, init_from=init_from)
    print(json.dumps(result, indent=2), flush=True)
