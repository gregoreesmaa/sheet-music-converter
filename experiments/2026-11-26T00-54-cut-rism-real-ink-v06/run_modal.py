"""Modal L4 launcher for Experiment 1: Geometry-Locked CUT-RISM Real-Ink Style Transfer.

Experiment ID: 2026-11-26T00-54-cut-rism-real-ink-v06
App Name:      exp1-cut-rism-real-ink-v06
GPU:           Modal L4 (24 GB VRAM, Ada Lovelace cc=8.9, native BF16)
Volume:        omr-cache mounted at /vol
Secret:        hf-secret (HF_TOKEN for private repo gregoreesmaa/sheet-music-converter-omr)

Usage (detached):
    .\\.venv\\Scripts\\modal.exe run --detach experiments/2026-11-26T00-54-cut-rism-real-ink-v06/run_modal.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tarfile
import threading
import time
from typing import Any, Dict, List, Optional

import modal

EXP_ID = "2026-11-26T00-54-cut-rism-real-ink-v06"
APP_NAME = "exp1-cut-rism-real-ink-v06"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"

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


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_extract_tar(tar_path: Path, dest_dir: Path) -> None:
    """Extract tar archive into dest_dir safely."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as tf:
        try:
            tf.extractall(path=dest_dir, filter="data")
        except TypeError:
            tf.extractall(path=dest_dir)


def _ensure_dataset_unpacked(
    api: Any,
    hf_filename: str,
    dest_dir: Path,
    marker_name: str = ".unpacked_ok",
) -> Path:
    """Download HF tar shard and unpack into /vol/datasets/<name> if not already cached."""
    marker = dest_dir / marker_name
    if marker.exists():
        print(f"[cache] Reusing already-unpacked dataset at {dest_dir}", flush=True)
        return dest_dir

    from huggingface_hub import hf_hub_download

    t0 = time.time()
    print(f"[cache] Downloading {hf_filename} from {HF_REPO_ID} ...", flush=True)
    local_tar = Path(
        hf_hub_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            filename=hf_filename,
            token=os.environ.get("HF_TOKEN"),
        )
    )
    print(
        f"[cache] Downloaded {hf_filename} ({local_tar.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.1f}s. Unpacking...",
        flush=True,
    )
    t1 = time.time()
    _safe_extract_tar(local_tar, dest_dir)
    marker.write_text(f"unpacked_at={_utc_now()}\nsource={hf_filename}\n", encoding="utf-8")
    vol.commit()
    print(f"[cache] Unpacked {hf_filename} -> {dest_dir} in {time.time() - t1:.1f}s (vol committed).", flush=True)
    return dest_dir


def _find_manifest(root: Path) -> Path:
    """Locate manifest.jsonl inside an unpacked hw_sim_v06 directory."""
    candidates = [
        root / "manifest.jsonl",
        root / "fullpage_manifest.jsonl",
    ]
    for c in candidates:
        if c.is_file():
            return c
    for p in sorted(root.rglob("manifest.jsonl")):
        return p
    for p in sorted(root.rglob("fullpage_manifest.jsonl")):
        return p
    raise FileNotFoundError(f"No manifest.jsonl found under {root}")


def _build_resolved_manifest(src_root: Path, src_manifest: Path, out_manifest: Path) -> int:
    """Resolve image paths (handling Windows/relative paths) to real paths under src_root."""
    img_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    by_rel: Dict[str, Path] = {}
    by_name: Dict[str, Path] = {}
    for p in sorted(src_root.rglob("*")):
        if p.is_file() and p.suffix.lower() in img_exts:
            rel = p.relative_to(src_root).as_posix()
            by_rel[rel] = p
            # Prefer files inside pages/ if duplicate names exist
            if p.name not in by_name or "/pages/" in f"/{rel}":
                by_name[p.name] = p

    resolved_rows: List[Dict[str, Any]] = []
    with open(src_manifest, "r", encoding="utf-8") as mf:
        for line in mf:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            raw_img = str(rec["image"]).replace("\\", "/")
            cand = Path(raw_img)
            resolved: Optional[Path] = None
            if cand.is_file():
                resolved = cand
            elif raw_img.lstrip("./") in by_rel:
                resolved = by_rel[raw_img.lstrip("./")]
            else:
                # Match suffix like pages/hwsim_00000.png or basename
                for tail_k in (2, 1):
                    parts = [pt for pt in raw_img.split("/") if pt]
                    if len(parts) >= tail_k:
                        suffix = "/".join(parts[-tail_k:])
                        if suffix in by_rel:
                            resolved = by_rel[suffix]
                            break
                if resolved is None:
                    base = raw_img.split("/")[-1]
                    resolved = by_name.get(base)
            if resolved is None or not resolved.is_file():
                raise FileNotFoundError(f"Could not resolve image '{rec['image']}' under {src_root}")
            new_rec = dict(rec)
            new_rec["_orig_image"] = rec["image"]
            new_rec["image"] = str(resolved)
            resolved_rows.append(new_rec)

    tmp_p = out_manifest.with_suffix(".jsonl.tmp")
    with open(tmp_p, "w", encoding="utf-8") as wf:
        for r in resolved_rows:
            wf.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp_p, out_manifest)
    return len(resolved_rows)


def _normalize_bekern_rel(raw_path: str) -> str:
    """Normalize a bekern path to a clean POSIX relative path (e.g. stage2_strips/foo.txt)."""
    norm = str(raw_path).replace("\\", "/")
    if "stage2_strips/" in norm:
        return "stage2_strips/" + norm.split("stage2_strips/", 1)[1]
    if not norm.startswith("/") and ":" not in norm:
        return norm.lstrip("./")
    return Path(norm).name


def _upload_status_json(api: Any, payload: Dict[str, Any]) -> None:
    """Upload run status ledger JSON to HF runs/<EXP_ID>.json."""
    tmp_json = Path(f"/tmp/{EXP_ID}_status.json")
    tmp_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    api.upload_file(
        path_or_fileobj=str(tmp_json),
        path_in_repo=f"runs/{EXP_ID}.json",
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        commit_message=f"exp1({EXP_ID}): status={payload.get('status', 'running')}",
    )


def _sync_checkpoints_to_hf(api: Any, ckpt_dir: Path, step_label: str) -> List[str]:
    """Upload latest checkpoints, train log, and latest preview PNG to HF."""
    uploaded: List[str] = []
    files_to_push = [
        ckpt_dir / "cut_rism_best.pt",
        ckpt_dir / "cut_rism_latest.pt",
        ckpt_dir / "train_log.jsonl",
    ]
    previews = sorted((ckpt_dir / "previews").glob("step_*.png")) if (ckpt_dir / "previews").is_dir() else []
    if previews:
        files_to_push.append(previews[-1])

    for fpath in files_to_push:
        if not fpath.is_file():
            continue
        rel = fpath.relative_to(ckpt_dir).as_posix()
        api.upload_file(
            path_or_fileobj=str(fpath),
            path_in_repo=f"checkpoints/{EXP_ID}/{rel}",
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"exp1({EXP_ID}): sync {rel} ({step_label})",
        )
        uploaded.append(rel)
    return uploaded


def _read_latest_log_metrics(log_path: Path) -> Dict[str, Any]:
    if not log_path.is_file():
        return {}
    last_line = ""
    with open(log_path, "r", encoding="utf-8") as lf:
        for line in lf:
            if line.strip():
                last_line = line.strip()
    if not last_line:
        return {}
    try:
        return json.loads(last_line)
    except json.JSONDecodeError:
        return {}


@app.function(
    image=image,
    gpu="L4",
    timeout=7200,
    volumes={"/vol": vol},
    secrets=[hf_secret],
)
def run_experiment(
    steps: int = 12000,
    batch_size: int = 4,
    crop_size: int = 512,
    lr: float = 2e-4,
    save_every: int = 500,
    preview_every: int = 500,
    log_every: int = 50,
    geom_lock_weight: float = 4.0,
    sobel_weight: float = 1.0,
    nce_weight: float = 1.0,
    gan_weight: float = 1.0,
    seed: int = 42,
    skip_train: bool = False,
) -> Dict[str, Any]:
    """Train CUT-RISM on L4, sync 5-min checkpoints, translate 500 hw_sim_v06 pages, upload tar."""
    sys.path.insert(0, "/app/mvd")

    from PIL import Image
    import torch
    from huggingface_hub import HfApi, hf_hub_download

    import train_cut_rism as cut

    Image.MAX_IMAGE_PIXELS = None
    t_start = time.time()

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    cc = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
    use_bf16 = bool(torch.cuda.is_available() and cc[0] >= 8)
    print(
        f"[exp1] {EXP_ID} starting on {device_name} (cc={cc[0]}.{cc[1]}, bf16={use_bf16})",
        flush=True,
    )

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN missing from Modal secret 'hf-secret'")
    api = HfApi(token=hf_token)

    # 1. Download & cache datasets onto Modal Volume /vol/datasets/
    datasets_root = Path("/vol/datasets")
    rism_dir = _ensure_dataset_unpacked(api, "datasets/rism_hw_5707.tar", datasets_root / "rism_hw_5707")
    hwsim_dir = _ensure_dataset_unpacked(api, "datasets/hw_sim_v06_500.tar", datasets_root / "hw_sim_v06_500")

    ckpt_dir = Path("/vol/ckpts") / EXP_ID
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path("/vol/out/cut_rism_v06")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Auto-resume from volume or HF if a previous checkpoint exists
    resume_ckpt: Optional[Path] = ckpt_dir / "cut_rism_latest.pt"
    if not resume_ckpt.is_file():
        try:
            dl_ckpt = hf_hub_download(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                filename=f"checkpoints/{EXP_ID}/cut_rism_latest.pt",
                token=hf_token,
            )
            resume_ckpt = Path(dl_ckpt)
            print(f"[exp1] Restored resume checkpoint from HF: {resume_ckpt}", flush=True)
        except Exception:
            resume_ckpt = None

    status_payload: Dict[str, Any] = {
        "exp_id": EXP_ID,
        "app_name": APP_NAME,
        "platform": "modal_l4",
        "gpu": device_name,
        "bf16": use_bf16,
        "status": "running_train",
        "started_at": _utc_now(),
        "updated_at": _utc_now(),
        "hyperparams": {
            "steps": steps,
            "batch_size": batch_size,
            "crop_size": crop_size,
            "lr": lr,
            "save_every": save_every,
            "geom_lock_weight": geom_lock_weight,
            "sobel_weight": sobel_weight,
            "nce_weight": nce_weight,
            "gan_weight": gan_weight,
            "seed": seed,
        },
        "hf_artifacts": {
            "checkpoints_prefix": f"checkpoints/{EXP_ID}/",
            "translated_tar": "datasets/cut_rism_v06_500.tar",
            "run_ledger": f"runs/{EXP_ID}.json",
        },
    }
    _upload_status_json(api, status_payload)

    # 2. Periodic 5-minute Modal Volume commit + 15-minute HF checkpoint sync thread
    stop_sync = threading.Event()

    def _sync_worker() -> None:
        last_hf_sync = time.time()
        while not stop_sync.wait(timeout=300.0):
            try:
                vol.commit()
                latest_metrics = _read_latest_log_metrics(ckpt_dir / "train_log.jsonl")
                step_num = latest_metrics.get("step", 0)
                print(f"[sync-5m] vol.commit() done at step={step_num}", flush=True)
                if time.time() - last_hf_sync >= 900.0:
                    pushed = _sync_checkpoints_to_hf(api, ckpt_dir, step_label=f"step_{step_num}")
                    status_payload["updated_at"] = _utc_now()
                    status_payload["latest_metrics"] = latest_metrics
                    _upload_status_json(api, status_payload)
                    last_hf_sync = time.time()
                    print(f"[sync-15m] HF uploaded {pushed} at step={step_num}", flush=True)
            except Exception as exc:
                print(f"[sync] Warning during periodic sync: {exc}", flush=True)

    sync_thread = threading.Thread(target=_sync_worker, daemon=True)
    sync_thread.start()

    # 3. Train Geometry-Locked CUT-RISM
    try:
        cfg = cut.CutRismConfig(
            ngf=64,
            ndf=64,
            n_blocks=9,
            gan_weight=gan_weight,
            nce_weight=nce_weight,
            geom_lock_weight=geom_lock_weight,
            sobel_weight=sobel_weight,
        )
        if not skip_train:
            best_ckpt = cut.run_train(
                src_dirs=[hwsim_dir],
                tgt_dirs=[rism_dir],
                out_dir=ckpt_dir,
                cfg=cfg,
                steps=steps,
                batch_size=batch_size,
                crop_size=crop_size,
                lr=lr,
                save_every=save_every,
                preview_every=preview_every,
                log_every=log_every,
                use_bf16=use_bf16,
                device_str="cuda",
                seed=seed,
                resume_path=resume_ckpt,
            )
        else:
            best_ckpt = ckpt_dir / "cut_rism_best.pt"
            if not best_ckpt.is_file():
                best_ckpt = ckpt_dir / "cut_rism_latest.pt"
            if not best_ckpt.is_file():
                raise FileNotFoundError(f"skip_train=True but no checkpoint found in {ckpt_dir}")
    finally:
        stop_sync.set()
        sync_thread.join(timeout=30.0)
        vol.commit()

    # Sync final training checkpoints to HF before translation
    latest_metrics = _read_latest_log_metrics(ckpt_dir / "train_log.jsonl")
    _sync_checkpoints_to_hf(api, ckpt_dir, step_label="train_complete")
    status_payload["status"] = "translating_500_pages"
    status_payload["updated_at"] = _utc_now()
    status_payload["train_elapsed_s"] = round(time.time() - t_start, 1)
    status_payload["latest_metrics"] = latest_metrics
    _upload_status_json(api, status_payload)

    # 4. Translate all 500 hw_sim_v06 pages with exact GT preservation
    src_manifest = _find_manifest(hwsim_dir)
    resolved_manifest = hwsim_dir / "_modal_resolved_manifest.jsonl"
    n_pages = _build_resolved_manifest(hwsim_dir, src_manifest, resolved_manifest)
    print(f"[exp1] Translating {n_pages} pages using {best_ckpt} ...", flush=True)

    t_tr = time.time()
    raw_out_manifest = cut.translate_manifest(
        checkpoint_path=best_ckpt,
        manifest_path=resolved_manifest,
        out_dir=out_dir,
        max_side=1536,
        tile_size=768,
        tile_overlap=96,
        use_bf16=use_bf16,
        device_str="cuda",
        resume=True,
    )
    vol.commit()

    # Rewrite manifest.jsonl with portable relative paths
    portable_rows: List[Dict[str, Any]] = []
    with open(raw_out_manifest, "r", encoding="utf-8") as rf:
        for line in rf:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out_img_p = Path(rec["image"])
            rec["image"] = f"pages/{out_img_p.name}"
            if "_orig_image" in rec:
                orig_name = str(rec.pop("_orig_image")).replace("\\", "/").split("/")[-1]
                rec["source_image"] = f"pages/{orig_name}"
            elif "source_image" in rec:
                src_name = str(rec["source_image"]).replace("\\", "/").split("/")[-1]
                rec["source_image"] = f"pages/{src_name}"
            if "bekerns" in rec and isinstance(rec["bekerns"], list):
                rec["bekerns"] = [_normalize_bekern_rel(b) for b in rec["bekerns"]]
            if "bekern" in rec and isinstance(rec["bekern"], str):
                rec["bekern"] = _normalize_bekern_rel(rec["bekern"])
            rec["style_transfer"] = "cut_rism"
            rec["style_transfer_exp"] = EXP_ID
            portable_rows.append(rec)

    portable_manifest = out_dir / "manifest.jsonl"
    tmp_manifest = out_dir / "manifest.jsonl.tmp"
    with open(tmp_manifest, "w", encoding="utf-8") as wf:
        for r in portable_rows:
            wf.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp_manifest, portable_manifest)
    vol.commit()
    print(
        f"[exp1] Translated {len(portable_rows)} pages in {time.time() - t_tr:.1f}s.",
        flush=True,
    )

    # 5. Pack cut_rism_v06_500.tar (manifest.jsonl + pages/*.png + stage2_strips/*.txt if present)
    tar_path = Path("/vol/out/cut_rism_v06_500.tar")
    tmp_tar = Path("/vol/out/cut_rism_v06_500.tar.tmp")
    n_txt = 0
    with tarfile.open(tmp_tar, "w") as tf:
        tf.add(portable_manifest, arcname="manifest.jsonl")
        for img_p in sorted((out_dir / "pages").glob("*.png")):
            tf.add(img_p, arcname=f"pages/{img_p.name}")
        # Include bekern GT text files if unpacked alongside hw_sim_v06_500
        strips_dirs = [p for p in hwsim_dir.rglob("stage2_strips") if p.is_dir()]
        if strips_dirs:
            for txt_p in sorted(strips_dirs[0].glob("*.txt")):
                tf.add(txt_p, arcname=f"stage2_strips/{txt_p.name}")
                n_txt += 1
    os.replace(tmp_tar, tar_path)
    vol.commit()

    tar_mb = tar_path.stat().st_size / 1e6
    print(
        f"[exp1] Packed {tar_path} ({tar_mb:.1f} MB, {len(portable_rows)} PNGs, {n_txt} bekern txts). Uploading to HF...",
        flush=True,
    )

    api.upload_file(
        path_or_fileobj=str(tar_path),
        path_in_repo="datasets/cut_rism_v06_500.tar",
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        commit_message=f"exp1({EXP_ID}): upload {len(portable_rows)} CUT-RISM translated hw_sim_v06 pages",
    )

    status_payload["status"] = "completed"
    status_payload["updated_at"] = _utc_now()
    status_payload["finished_at"] = _utc_now()
    status_payload["total_elapsed_s"] = round(time.time() - t_start, 1)
    status_payload["translated_pages"] = len(portable_rows)
    status_payload["bekern_gt_files"] = n_txt
    status_payload["tar_size_mb"] = round(tar_mb, 2)
    _upload_status_json(api, status_payload)

    print(f"[exp1] DONE! Uploaded datasets/cut_rism_v06_500.tar and runs/{EXP_ID}.json", flush=True)
    return status_payload


@app.local_entrypoint()
def main(
    steps: int = 12000,
    batch_size: int = 4,
    save_every: int = 500,
    skip_train: bool = False,
) -> None:
    call = run_experiment.spawn(
        steps=steps,
        batch_size=batch_size,
        save_every=save_every,
        skip_train=skip_train,
    )
    print(f"[exp1] Spawned detached call_id={call.object_id}", flush=True)
