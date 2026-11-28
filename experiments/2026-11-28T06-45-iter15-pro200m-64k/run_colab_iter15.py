#!/usr/bin/env python3
"""Google Colab H100/A100/L4 Recoverable Trainer for Iter 15 Flagship (`iter15_pro200m`, 64k).

Experiment ID: 2026-11-28T06-45-iter15-pro200m-64k
Architecture:  UniOMRv2Net `iter15_pro200m` (d_model=768, 194.14M infer, 120.41 MB Hybrid INT4)
Data Mix:      `--stratified-preset v4_hw70` (30% Printed / 69.5% HW / 0.5% Neg)
Key Upgrades:  YOCO-3Bank + CLA-2:1 + ResFormer v_res + DiffCrossAttn + CoordKey-CA +
               ModeGate + Chained MTP-4 + QuantVLA head_temp + BD3-LM repair (w=0.25, n_views=1) +
               1536px multiscale + fixed HxW buckets + RISM FDA/AdaIN + Muon+AdamW WSD + Late INT4 QAT.

Checkpointing & Recovery Guarantees:
  - Auto-resumes from `checkpoints/iter15_pro200m_64k/latest/` on HF Private Hub
    (`gregoreesmaa/sheet-music-converter-omr`) via true `train_uni_v2.py --resume <latest_ckpt>`
    with fixed `--steps 64000`, preserving WSD cooldown, CASS, QAT, EMA, and Muon/AdamW states
    seamlessly across Colab preemptions.
  - Background sync thread pushes rolling `uni_omr_v2_latest.pt` + `uni_omr_v2_ema_latest.pt`
    (+ `train_log.json` + `run_state.json`) every 5 minutes (atomic snapshot copy, mtime dedup,
    exponential backoff) AND forces a final flush in `finally:`.
  - Permanently archives exact EMA milestone files (`uni_omr_v2_ema_step{m:06d}.pt`) to:
      `checkpoints/iter15_pro200m_64k/milestones/step_XXXXXX_ema.pt`
    at global steps `1000, 2000, 4000, 8000, 16000, 32000, 64000` (never backfills from latest).
  - Supports `--smoke-steps N` (e.g. `--smoke-steps 20`) for fast pre-launch GPU verification
    and `--dry-run` for CPU self-test without touching GPU or HF network.
"""

from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = (
    os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") or "1"
)

EXP_ID = "2026-11-28T06-45-iter15-pro200m-64k"
RUN_TAG = "iter15_pro200m_64k"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_CKPT_ROOT = f"checkpoints/{RUN_TAG}"
HF_LATEST_PREFIX = f"{HF_CKPT_ROOT}/latest"
HF_MILESTONES_PREFIX = f"{HF_CKPT_ROOT}/milestones"
HF_RUN_LEDGER_PATH = f"runs/{EXP_ID}.json"
HF_ITER15_DATA_PREFIX = "datasets/iter15_v2a/"
HF_ITER15_SHARDS_PREFIX = "datasets/iter15_v2a/shards/"
HF_ITER15_MANIFEST_PATH = f"{HF_ITER15_SHARDS_PREFIX}uni_fullpage_iter15_v2a.jsonl"
HF_ITER15_NUM_SHARDS = 14
HF_ITER15_SHARD_FILES: Tuple[str, ...] = tuple(
    f"{HF_ITER15_SHARDS_PREFIX}iter15_shard_{i:02d}.tar"
    for i in range(HF_ITER15_NUM_SHARDS)
)
HF_TEACHER_CKPT_PATH = "checkpoints/iter14_v2a_stage2_16k/uni_omr_v2_ema_latest.pt"
HF_DL_CACHE_DIR = Path("/tmp/hf_dl_cache")

HF_CORE_DATASET_SHARDS: Tuple[Tuple[str, str], ...] = (
    ("datasets/rism_hw_5707.tar", "unlabeled-10k-real/rism_hw"),
)

BUNDLE_CANDIDATES: Tuple[Path, ...] = (
    Path("/content/iter15_bundle.tar.gz"),
    Path("/content/ov_mvd_bundle.tar.gz"),
    Path("/content/ov_bundle.tar.gz"),
    Path("/content/bundle.tar.gz"),
    Path("/content/repo_bundle.tar.gz"),
)

DEFAULT_DATA_ROOT = Path("/content/data")
DEFAULT_RUN_ROOT = Path("/content/runs/iter15_pro200m_64k")
HF_TOKEN_FILE = Path("/tmp/.hf_token")

TARGET_STEPS_DEFAULT = 64000
MILESTONE_STEPS: Tuple[int, ...] = (1000, 2000, 4000, 8000, 16000, 32000, 64000)
SAVE_EVERY_MINUTES = 5.0
SYNC_INTERVAL_SECONDS = 300.0

ROLLING_FILENAMES: Tuple[str, ...] = (
    "uni_omr_v2_latest.pt",
    "uni_omr_v2_ema_latest.pt",
    "uni_omr_v2_ema_best.pt",
    "train_log.json",
    "run_state.json",
)

_EMA_STEP_RE = re.compile(r"^uni_omr_v2_ema_step(\d+)\.pt$")
_WIN_DRIVE_TMP_RE = re.compile(r"^[A-Za-z]:[/\\]+tmp[/\\]+", re.IGNORECASE)
_WIN_REPO_RE = re.compile(
    r"^[A-Za-z]:[/\\]+.*?sheet-music-converter[/\\]+", re.IGNORECASE
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(msg: str) -> None:
    print(f"[{RUN_TAG} {_utc_now()}] {msg}", flush=True)


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _read_hf_token(*, required: bool = True) -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token and HF_TOKEN_FILE.exists():
        try:
            os.chmod(HF_TOKEN_FILE, 0o600)
        except OSError:
            pass
        token = HF_TOKEN_FILE.read_text(encoding="utf-8").strip()
    if not token:
        cache_tok = Path.home() / ".cache" / "huggingface" / "token"
        if cache_tok.is_file():
            token = cache_tok.read_text(encoding="utf-8").strip()
    if not token and required:
        raise RuntimeError(
            "HF token not found in HF_TOKEN env var, /tmp/.hf_token, or ~/.cache/huggingface/token."
        )
    return token


def _purge_tmp_token_file() -> None:
    if HF_TOKEN_FILE.exists():
        try:
            HF_TOKEN_FILE.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# 1. GPU Tier Auto-Tuning for `iter15_pro200m` @ 1536px (effective batch = 16)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class GpuTuningProfile:
    tier: str
    device_name: str
    vram_gb: float
    batch_size: int
    grad_accum: int
    use_grad_checkpoint: bool
    use_bf16: bool = True
    use_no_cudnn: bool = False  # Linux Colab cuDNN + fixed HxW buckets is fast (H100/A100 use --channels-last NHWC)
    use_channels_last: bool = False
    prefetch_threads: int = 6
    num_workers: int = 4


def select_gpu_profile(
    *,
    vram_gb: Optional[float] = None,
    device_name: Optional[str] = None,
) -> GpuTuningProfile:
    """Auto-tune batch_size / grad_accum for `iter15_pro200m` (194M) at 1536px."""
    if vram_gb is None:
        try:
            import torch

            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                vram_gb = float(props.total_memory) / float(1024**3)
                device_name = device_name or str(props.name)
            else:
                vram_gb = 0.0
                device_name = device_name or "CPU"
        except Exception:
            vram_gb = 0.0
            device_name = device_name or "CPU"

    is_linux = sys.platform.startswith("linux")
    dev_name = device_name or f"GPU-{vram_gb:.1f}GB"
    if vram_gb >= 70.0:
        return GpuTuningProfile(
            tier="H100-80GB",
            device_name=dev_name,
            vram_gb=round(vram_gb, 2),
            batch_size=8,
            grad_accum=2,
            use_grad_checkpoint=True,
            use_bf16=True,
            use_no_cudnn=False,
            use_channels_last=False,  # fused-Adam resume crash on Colab w/ channels_last
            prefetch_threads=8,
            num_workers=8,
        )
    if vram_gb >= 36.0:
        return GpuTuningProfile(
            tier="A100-40GB",
            device_name=dev_name,
            vram_gb=round(vram_gb, 2),
            batch_size=4,
            grad_accum=4,
            use_grad_checkpoint=True,
            use_bf16=True,
            use_no_cudnn=False,
            use_channels_last=False,  # fused-Adam resume crash on Colab w/ channels_last
            prefetch_threads=6,
            num_workers=6,
        )
    return GpuTuningProfile(
        tier="L4-24GB",
        device_name=dev_name,
        vram_gb=round(vram_gb, 2),
        batch_size=1,
        grad_accum=16,
        use_grad_checkpoint=True,
        use_bf16=True,
        use_no_cudnn=False,
        use_channels_last=False,
        prefetch_threads=4,
        num_workers=2,
    )


# ---------------------------------------------------------------------------
# 2. Windows -> Linux Manifest Path Remapping
# ---------------------------------------------------------------------------


def remap_windows_path(
    raw_path: str,
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    repo_root: Optional[Path] = None,
) -> str:
    if not raw_path:
        return raw_path
    s = raw_path.strip()
    if not s:
        return s
    norm = s.replace("\\", "/")

    m_tmp = _WIN_DRIVE_TMP_RE.match(s)
    if m_tmp:
        rel = norm[m_tmp.end() :].lstrip("/")
        return (data_root / rel).as_posix()

    m_repo = _WIN_REPO_RE.match(s)
    if m_repo:
        rel = norm[m_repo.end() :].lstrip("/")
        base = repo_root if repo_root is not None else data_root
        return (base / rel).as_posix()

    if norm.startswith("/"):
        return norm

    if repo_root is not None and norm.startswith("tmp-train/"):
        return (repo_root / norm).as_posix()

    return (data_root / norm).as_posix()


def remap_jsonl_manifest(
    src_manifest: Path,
    dst_manifest: Path,
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    repo_root: Optional[Path] = None,
    drop_missing: bool = False,
) -> Dict[str, int]:
    total = 0
    kept = 0
    remapped_fields = 0
    missing_images = 0
    out_lines: List[str] = []

    for raw_line in src_manifest.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        total += 1
        rec = json.loads(line)

        for key in ("image", "img", "image_path", "bekern", "bekern_path", "musicxml", "src_path"):
            if key in rec and isinstance(rec[key], str) and rec[key]:
                new_val = remap_windows_path(
                    rec[key], data_root=data_root, repo_root=repo_root
                )
                if new_val != rec[key]:
                    remapped_fields += 1
                rec[key] = new_val

        for list_key in ("bekerns", "bekern_paths", "targets"):
            if list_key in rec and isinstance(rec[list_key], list):
                new_list = []
                for item in rec[list_key]:
                    if isinstance(item, str):
                        nv = remap_windows_path(
                            item, data_root=data_root, repo_root=repo_root
                        )
                        if nv != item:
                            remapped_fields += 1
                        new_list.append(nv)
                    else:
                        new_list.append(item)
                rec[list_key] = new_list

        img_p = rec.get("image") or rec.get("img") or rec.get("image_path")
        if img_p and not Path(img_p).is_file():
            missing_images += 1
            if drop_missing:
                continue

        kept += 1
        out_lines.append(json.dumps(rec, ensure_ascii=False))

    _atomic_write_text(dst_manifest, ("\n".join(out_lines) + "\n") if out_lines else "")
    return {
        "total_rows": total,
        "kept_rows": kept,
        "remapped_fields": remapped_fields,
        "missing_images": missing_images,
    }


def _read_train_log_step(log_path: Path) -> int:
    if not log_path.is_file():
        return 0
    try:
        data = json.loads(log_path.read_text(encoding="utf-8"))
        if isinstance(data, list) and data:
            for rec in reversed(data):
                if isinstance(rec, dict) and rec.get("step") is not None:
                    return int(rec["step"])
        elif isinstance(data, dict):
            if data.get("step") is not None:
                return int(data["step"])
            hist = data.get("history")
            if isinstance(hist, list) and hist:
                for rec in reversed(hist):
                    if isinstance(rec, dict) and rec.get("step") is not None:
                        return int(rec["step"])
    except Exception:
        pass
    return 0


def _stage_snapshot(src_path: Path, dst_path: Path) -> int:
    """Zero-copy hardlink staging on same filesystem, falling back to shutil.copy2."""
    if dst_path.exists():
        dst_path.unlink()
    try:
        os.link(src_path, dst_path)
    except OSError:
        shutil.copy2(src_path, dst_path)
    return int(dst_path.stat().st_size)


def _read_ckpt_step(ckpt_path: Path) -> int:
    try:
        import torch

        try:
            ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        except Exception:
            ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if isinstance(ck, dict):
            return int(ck.get("step", 0) or 0)
    except Exception:
        pass
    return 0


def milestone_repo_path(step: int, milestones_prefix: str = HF_MILESTONES_PREFIX) -> str:
    return f"{milestones_prefix.rstrip('/')}/step_{int(step):06d}_ema.pt"


# ---------------------------------------------------------------------------
# 3. Background 5-Min Rolling HF Syncer + Permanent EMA Milestone Archiver
# ---------------------------------------------------------------------------


class HFCheckpointAndMilestoneSyncer:
    """Non-blocking background syncer for rolling latest ckpts + permanent EMA milestones.

    - Every `interval_seconds` (default 300s = 5 min), uploads modified rolling files
      (`uni_omr_v2_latest.pt`, `uni_omr_v2_ema_latest.pt`, `uni_omr_v2_ema_best.pt`,
      `train_log.json`, `run_state.json`) to `checkpoints/iter15_pro200m_64k/latest/`.
    - Archives ONLY exact `uni_omr_v2_ema_step{m:06d}.pt` milestone files produced by
      `train_uni_v2.py --ema-milestone-steps ...` to
      `checkpoints/iter15_pro200m_64k/milestones/step_{m:06d}_ema.pt` (never backfills
      milestones from `uni_omr_v2_ema_latest.pt`).
    - Uses atomic `.tmp` snapshot copies (guarded against concurrent writer renames) and
      retries transient network errors with exponential backoff.
    """

    def __init__(
        self,
        *,
        out_dir: Path,
        hf_repo_id: str = HF_REPO_ID,
        latest_prefix: str = HF_LATEST_PREFIX,
        milestones_prefix: str = HF_MILESTONES_PREFIX,
        hf_token: str = "",
        initial_step: int = 0,
        milestone_steps: Sequence[int] = MILESTONE_STEPS,
        already_uploaded_milestones: Sequence[int] = (),
        archive_milestones: bool = True,
        interval_seconds: float = SYNC_INTERVAL_SECONDS,
        max_retries: int = 5,
        base_backoff_s: float = 2.0,
        dry_run: bool = False,
        upload_fn: Optional[Callable[[Path, str, str], None]] = None,
        ckpt_step_reader: Optional[Callable[[Path], int]] = None,
    ) -> None:
        self.out_dir = out_dir
        self.hf_repo_id = hf_repo_id
        self.latest_prefix = latest_prefix.rstrip("/")
        self.milestones_prefix = milestones_prefix.rstrip("/")
        self.hf_token = hf_token
        self.milestone_steps = tuple(sorted(set(int(s) for s in milestone_steps)))
        self.uploaded_milestones: Set[int] = set(int(s) for s in already_uploaded_milestones)
        self.archive_milestones = bool(archive_milestones)
        self.interval_seconds = max(0.05, float(interval_seconds))
        self.max_retries = max(1, int(max_retries))
        self.base_backoff_s = max(0.01, float(base_backoff_s))
        self.dry_run = dry_run
        self._upload_fn = upload_fn
        self._ckpt_step_reader = ckpt_step_reader or _read_ckpt_step

        self._synced_meta: Dict[str, Tuple[int, int]] = {}
        self._train_log_step_cache: Tuple[Tuple[int, int], int] = ((0, 0), 0)
        self._latest_ema_step_cache: Tuple[Tuple[int, int], int] = ((0, 0), 0)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.upload_history: List[Dict[str, Any]] = []
        self.error_count = 0
        self.last_seen_global_step = max(0, int(initial_step))

    def _upload_with_backoff(self, src_path: Path, path_in_repo: str, commit_msg: str) -> bool:
        stage_dir = self.out_dir / ".sync_stage"
        stage_dir.mkdir(parents=True, exist_ok=True)
        tmp_snap = stage_dir / f"{src_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        try:
            try:
                size_bytes = _stage_snapshot(src_path, tmp_snap)
            except OSError as exc:
                self.error_count += 1
                _log(f"[HF-Sync WARN] Snapshot staging failed for {src_path.name}: {exc}")
                return False
            for attempt in range(1, self.max_retries + 1):
                try:
                    if self._upload_fn is not None:
                        self._upload_fn(tmp_snap, path_in_repo, commit_msg)
                    elif self.dry_run:
                        _log(
                            f"[HF-Sync DRY-RUN] would upload {src_path.name} "
                            f"({size_bytes} B) -> {path_in_repo}"
                        )
                    else:
                        from huggingface_hub import HfApi

                        api = HfApi(token=self.hf_token)
                        api.upload_file(
                            path_or_fileobj=str(tmp_snap),
                            path_in_repo=path_in_repo,
                            repo_id=self.hf_repo_id,
                            repo_type="dataset",
                            commit_message=commit_msg,
                        )
                    self.upload_history.append(
                        {
                            "local_path": str(src_path),
                            "path_in_repo": path_in_repo,
                            "size_bytes": size_bytes,
                            "utc": _utc_now(),
                            "dry_run": self.dry_run,
                        }
                    )
                    return True
                except Exception as exc:
                    self.error_count += 1
                    if attempt >= self.max_retries:
                        _log(
                            f"[HF-Sync WARN] Failed uploading {path_in_repo} after "
                            f"{self.max_retries} attempts: {exc}"
                        )
                        return False
                    sleep_s = self.base_backoff_s * (2 ** (attempt - 1))
                    _log(
                        f"[HF-Sync RETRY] Upload {path_in_repo} attempt {attempt}/{self.max_retries} "
                        f"failed ({exc}); retrying in {sleep_s:.1f}s..."
                    )
                    time.sleep(sleep_s)
        finally:
            try:
                if tmp_snap.exists():
                    tmp_snap.unlink()
            except OSError:
                pass
        return False

    def _upload_batch_with_backoff(
        self,
        items: Sequence[Tuple[Path, str, Tuple[int, int]]],
        *,
        reason: str,
    ) -> List[Tuple[Path, str, Tuple[int, int]]]:
        """Batch-upload rolling files in a single HF create_commit (with per-file fallback)."""
        if not items:
            return []

        if self._upload_fn is not None or len(items) == 1:
            succeeded: List[Tuple[Path, str, Tuple[int, int]]] = []
            for local_path, repo_path, sig in items:
                commit_msg = f"ckpt({RUN_TAG}): {reason} sync {local_path.name}"
                if self._upload_with_backoff(local_path, repo_path, commit_msg):
                    succeeded.append((local_path, repo_path, sig))
            return succeeded

        stage_dir = self.out_dir / ".sync_stage"
        stage_dir.mkdir(parents=True, exist_ok=True)
        staged: List[Tuple[Path, str, Tuple[int, int], Path, int]] = []
        pid = os.getpid()
        tid = threading.get_ident()
        try:
            for idx, (local_path, repo_path, sig) in enumerate(items):
                tmp_snap = stage_dir / f"{local_path.name}.{pid}.{tid}.{idx}.tmp"
                try:
                    size_bytes = _stage_snapshot(local_path, tmp_snap)
                    staged.append((local_path, repo_path, sig, tmp_snap, size_bytes))
                except OSError as exc:
                    self.error_count += 1
                    _log(f"[HF-Sync WARN] Snapshot staging failed for {local_path.name}: {exc}")

            if not staged:
                return []

            names_summary = ", ".join(p.name for p, _, _, _, _ in staged)
            commit_msg = (
                f"ckpt({RUN_TAG}): {reason} sync ({len(staged)} files: {names_summary})"
            )

            if self.dry_run:
                for local_path, repo_path, sig, _tmp_snap, size_bytes in staged:
                    _log(
                        f"[HF-Sync DRY-RUN] would batch-commit {local_path.name} "
                        f"({size_bytes} B) -> {repo_path}"
                    )
                    self.upload_history.append(
                        {
                            "local_path": str(local_path),
                            "path_in_repo": repo_path,
                            "size_bytes": size_bytes,
                            "utc": _utc_now(),
                            "dry_run": True,
                        }
                    )
                return [(p, rp, sig) for p, rp, sig, _, _ in staged]

            for attempt in range(1, self.max_retries + 1):
                try:
                    from huggingface_hub import CommitOperationAdd, HfApi

                    api = HfApi(token=self.hf_token)
                    operations = [
                        CommitOperationAdd(
                            path_in_repo=repo_path,
                            path_or_fileobj=str(tmp_snap),
                        )
                        for _, repo_path, _, tmp_snap, _ in staged
                    ]
                    api.create_commit(
                        repo_id=self.hf_repo_id,
                        repo_type="dataset",
                        operations=operations,
                        commit_message=commit_msg,
                    )
                    now_utc = _utc_now()
                    for local_path, repo_path, _sig, _tmp_snap, size_bytes in staged:
                        self.upload_history.append(
                            {
                                "local_path": str(local_path),
                                "path_in_repo": repo_path,
                                "size_bytes": size_bytes,
                                "utc": now_utc,
                                "dry_run": False,
                            }
                        )
                    return [(p, rp, sig) for p, rp, sig, _, _ in staged]
                except Exception as exc:
                    self.error_count += 1
                    if attempt >= self.max_retries:
                        _log(
                            f"[HF-Sync WARN] Batch create_commit failed after "
                            f"{self.max_retries} attempts ({exc}); falling back to per-file upload_file."
                        )
                        break
                    sleep_s = self.base_backoff_s * (2 ** (attempt - 1))
                    _log(
                        f"[HF-Sync RETRY] Batch commit attempt {attempt}/{self.max_retries} "
                        f"failed ({exc}); retrying in {sleep_s:.1f}s..."
                    )
                    time.sleep(sleep_s)
        finally:
            for _, _, _, tmp_snap, _ in staged:
                try:
                    if tmp_snap.exists():
                        tmp_snap.unlink()
                except OSError:
                    pass

        # Fallback to per-file upload_file
        succeeded = []
        for local_path, repo_path, sig in items:
            commit_msg = f"ckpt({RUN_TAG}): {reason} sync {local_path.name}"
            if self._upload_with_backoff(local_path, repo_path, commit_msg):
                succeeded.append((local_path, repo_path, sig))
        return succeeded

    def _update_run_state_file(self) -> None:
        state_path = self.out_dir / "run_state.json"
        payload: Dict[str, Any] = {}
        if state_path.is_file():
            try:
                loaded = json.loads(state_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    payload.update(loaded)
            except Exception:
                pass
        payload.update(
            {
                "exp_id": EXP_ID,
                "run_tag": RUN_TAG,
                "last_global_step": self.last_seen_global_step,
                "uploaded_milestones": sorted(self.uploaded_milestones),
                "updated_utc": _utc_now(),
            }
        )
        _atomic_write_json(state_path, payload)

    def _refresh_last_seen_step(self) -> None:
        # Fast path: read latest step from lightweight train_log.json (<1 ms)
        train_log = self.out_dir / "train_log.json"
        if train_log.is_file():
            try:
                st = train_log.stat()
                sig = (int(st.st_mtime_ns), int(st.st_size))
                if self._train_log_step_cache[0] == sig:
                    log_step = self._train_log_step_cache[1]
                else:
                    log_step = _read_train_log_step(train_log)
                    self._train_log_step_cache = (sig, log_step)
                if log_step > 0:
                    if log_step > self.last_seen_global_step:
                        self.last_seen_global_step = log_step
                    return
            except OSError:
                pass

        # Fallback: inspect EMA checkpoint only when train_log.json has no steps yet
        ema_latest = self.out_dir / "uni_omr_v2_ema_latest.pt"
        if not ema_latest.is_file():
            return
        try:
            st = ema_latest.stat()
            sig = (int(st.st_mtime_ns), int(st.st_size))
            if self._latest_ema_step_cache[0] == sig:
                ckpt_step = self._latest_ema_step_cache[1]
            else:
                ckpt_step = int(self._ckpt_step_reader(ema_latest))
                self._latest_ema_step_cache = (sig, ckpt_step)
            if ckpt_step > self.last_seen_global_step:
                self.last_seen_global_step = ckpt_step
        except OSError:
            pass

    def _detect_and_upload_milestones(self, reason: str) -> int:
        if not self.archive_milestones or not self.out_dir.is_dir():
            return 0
        uploaded = 0

        # Upload ONLY exact `uni_omr_v2_ema_step{m:06d}.pt` milestone files saved by train_uni_v2.py
        for target_m in self.milestone_steps:
            if target_m in self.uploaded_milestones:
                continue
            entry = self.out_dir / f"uni_omr_v2_ema_step{target_m:06d}.pt"
            if not entry.is_file():
                continue
            if target_m > self.last_seen_global_step:
                self.last_seen_global_step = target_m
            repo_path = milestone_repo_path(target_m, self.milestones_prefix)
            msg = f"milestone({RUN_TAG}): step {target_m} EMA ({reason})"
            if self._upload_with_backoff(entry, repo_path, msg):
                self.uploaded_milestones.add(target_m)
                uploaded += 1
                _log(f"[HF-Milestone] Archived permanent EMA milestone -> {repo_path}")

        self._refresh_last_seen_step()
        return uploaded

    def sync_once(self, *, reason: str = "periodic") -> int:
        with self._lock:
            uploaded = 0
            uploaded += self._detect_and_upload_milestones(reason=reason)
            self._update_run_state_file()

            if self.out_dir.is_dir():
                pending: List[Tuple[Path, str, Tuple[int, int]]] = []
                for fname in ROLLING_FILENAMES:
                    local_path = self.out_dir / fname
                    if not local_path.is_file():
                        continue
                    try:
                        st = local_path.stat()
                    except OSError:
                        continue
                    repo_path = f"{self.latest_prefix}/{fname}"
                    sig = (int(st.st_mtime_ns), int(st.st_size))
                    if self._synced_meta.get(repo_path) == sig:
                        continue
                    pending.append((local_path, repo_path, sig))

                if pending:
                    synced_items = self._upload_batch_with_backoff(pending, reason=reason)
                    for _local_path, repo_path, sig in synced_items:
                        self._synced_meta[repo_path] = sig
                        uploaded += 1

            if uploaded > 0:
                _log(
                    f"[HF-Sync] Synced {uploaded} file(s) ({reason}) | "
                    f"global_step={self.last_seen_global_step} | "
                    f"milestones={sorted(self.uploaded_milestones)}"
                )
            return uploaded

    def _run_loop(self) -> None:
        _log(
            f"[HF-Sync] Background 5-min sync thread started "
            f"(interval={self.interval_seconds:.1f}s, latest={self.latest_prefix}/, "
            f"milestones={self.milestone_steps})"
        )
        while not self._stop_event.wait(self.interval_seconds):
            try:
                self.sync_once(reason="5min-periodic")
            except Exception as exc:
                self.error_count += 1
                _log(f"[HF-Sync WARN] Unexpected error in periodic loop: {exc}")

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, name="hf-iter15-syncer", daemon=True
        )
        self._thread.start()

    def stop_and_final_sync(self) -> int:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=30.0)
        return self.sync_once(reason="final-flush")


# ---------------------------------------------------------------------------
# 4. Repo / Dependency / HF Dataset & Auto-Resume Helpers
# ---------------------------------------------------------------------------


def _run_cmd(
    cmd: Sequence[str] | str,
    *,
    shell: bool = False,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
) -> None:
    cmd_disp = cmd if isinstance(cmd, str) else " ".join(str(c) for c in cmd)
    _log(f"$ {cmd_disp}")
    proc = subprocess.Popen(
        cmd,
        shell=shell,
        cwd=str(cwd) if cwd is not None else None,
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
        raise RuntimeError(f"Command failed (rc={rc}): {cmd_disp}")


def _resolve_repo_root() -> Path:
    here = Path(__file__).resolve()
    candidates = [
        here.parents[2] if len(here.parents) >= 3 else Path.cwd(),
        Path("/workspace/sheet-music-converter"),
        Path("/content/sheet-music-converter"),
        Path("/content/workspace/sheet-music-converter"),
        Path.cwd(),
    ]
    for c in candidates:
        if (c / "tmp-train" / "mvd" / "train_uni_v2.py").is_file():
            return c

    for bundle in BUNDLE_CANDIDATES:
        if bundle.is_file():
            extract_root = Path("/content")
            extract_root.mkdir(parents=True, exist_ok=True)
            _log(f"Extracting repo bundle {bundle} -> {extract_root}...")
            with tarfile.open(bundle, "r:*") as tf:
                tf.extractall(path=extract_root)
            for c in (extract_root / "sheet-music-converter", extract_root):
                if (c / "tmp-train" / "mvd" / "train_uni_v2.py").is_file():
                    return c

    raise FileNotFoundError(
        "Could not locate tmp-train/mvd/train_uni_v2.py in repo or uploaded bundle."
    )


def _ensure_python_deps() -> None:
    pkgs: List[str] = []
    for mod, pkg in (
        ("huggingface_hub", "huggingface_hub>=0.26"),
        ("hf_transfer", "hf_transfer"),
        ("PIL", "pillow"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("tqdm", "tqdm"),
    ):
        try:
            __import__(mod)
        except ImportError:
            pkgs.append(pkg)
    if pkgs:
        _run_cmd([sys.executable, "-m", "pip", "install", "-q", *pkgs])
        importlib.invalidate_caches()


def _extract_tar_to(tar_path: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    tar_bin = shutil.which("tar")
    if tar_bin:
        try:
            subprocess.run(
                [tar_bin, "-xf", str(tar_path), "-C", str(dst_dir)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            return
        except Exception as exc:
            _log(
                f"[WARN] system tar -xf failed for {tar_path.name} ({exc}); "
                "falling back to Python tarfile."
            )
    with tarfile.open(tar_path, "r:*") as tf:
        tf.extractall(path=dst_dir)


def _download_hf_file(hf_token: str, repo_filename: str, dst_path: Path) -> Path:
    from huggingface_hub import hf_hub_download

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    HF_DL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        dl = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=repo_filename,
            repo_type="dataset",
            token=hf_token,
            local_dir=str(HF_DL_CACHE_DIR),
        )
        shutil.move(str(dl), str(dst_path))
    finally:
        shutil.rmtree(HF_DL_CACHE_DIR, ignore_errors=True)
    return dst_path


def prepare_dataset(
    *,
    hf_token: str,
    data_root: Path,
    repo_root: Path,
    manifest_name: str = "uni_fullpage_iter15_v2a.jsonl",
    smoke_mode: bool = False,
) -> Dict[str, Any]:
    """Stream HF 1536px JPEG Q95 shards (`datasets/iter15_v2a/shards/*` + RISM) onto Colab NVMe.

    - Full mode: downloads `uni_fullpage_iter15_v2a.jsonl`, all 14 shards
      (`iter15_shard_00.tar` .. `iter15_shard_13.tar`, 55,015 pages, 17.83 GB),
      and `datasets/rism_hw_5707.tar`.
    - Smoke mode: downloads `uni_fullpage_iter15_v2a.jsonl` and ONLY
      `iter15_shard_00.tar` (3,930 rows) so smoke staging completes in <30s.
    - Deletes each `.tar` immediately after `tar -xf` extraction to keep peak
      Colab disk around ~20 GB.
    - Rewrites relative `pages/shard_XX/...` and `bekerns/shard_XX/...` paths
      to absolute `f"{data_root}/..."` paths, keeping only rows whose extracted
      image exists on disk.
    """
    data_root.mkdir(parents=True, exist_ok=True)

    # 1. Download manifest JSONL
    manifest_repo_path = f"{HF_ITER15_SHARDS_PREFIX}{manifest_name}"
    src_manifest = data_root / manifest_name
    if not (src_manifest.is_file() and src_manifest.stat().st_size > 0):
        _log(f"Downloading manifest {manifest_repo_path} -> {src_manifest}...")
        _download_hf_file(hf_token, manifest_repo_path, src_manifest)

    # 2. Stream & extract iter15 shards one-by-one, deleting each .tar immediately
    shard_files = HF_ITER15_SHARD_FILES[:1] if smoke_mode else HF_ITER15_SHARD_FILES
    _log(
        f"Staging {len(shard_files)}/{HF_ITER15_NUM_SHARDS} iter15_v2a shard(s) "
        f"(smoke_mode={smoke_mode}) -> {data_root}..."
    )
    for rf in shard_files:
        shard_name = Path(rf).name
        marker = data_root / ".unpacked" / f"iter15_v2a_shards__{shard_name}.done"
        if marker.is_file():
            continue
        tmp_dst = data_root / ".downloads" / shard_name
        _log(f"Downloading {rf}...")
        _download_hf_file(hf_token, rf, tmp_dst)
        try:
            _log(f"Extracting {shard_name} -> {data_root}...")
            _extract_tar_to(tmp_dst, data_root)
        finally:
            try:
                if tmp_dst.exists():
                    tmp_dst.unlink()
            except OSError:
                pass
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(_utc_now() + "\n", encoding="utf-8")

    # 3. Download & extract RISM style pool (full mode only)
    if not smoke_mode:
        for rf, target_subdir in HF_CORE_DATASET_SHARDS:
            marker = data_root / ".unpacked" / f"{Path(rf).name}.done"
            if marker.is_file():
                continue
            dst_dir = data_root / target_subdir
            tmp_dst = data_root / ".downloads" / Path(rf).name
            _log(f"Downloading core shard {rf} -> {dst_dir}...")
            _download_hf_file(hf_token, rf, tmp_dst)
            try:
                _log(f"Extracting {Path(rf).name} -> {dst_dir}...")
                _extract_tar_to(tmp_dst, dst_dir)
            finally:
                try:
                    if tmp_dst.exists():
                        tmp_dst.unlink()
                except OSError:
                    pass
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(_utc_now() + "\n", encoding="utf-8")

    # 4. Rewrite relative shard paths -> absolute data_root paths (drop missing images)
    dst_manifest = data_root / (
        f"{src_manifest.stem}_colab_smoke.jsonl"
        if smoke_mode
        else f"{src_manifest.stem}_colab.jsonl"
    )
    stats = remap_jsonl_manifest(
        src_manifest,
        dst_manifest,
        data_root=data_root,
        repo_root=repo_root,
        drop_missing=True,
    )
    _log(f"Remapped manifest {src_manifest.name} -> {dst_manifest.name}: {stats}")
    if stats["kept_rows"] == 0:
        raise RuntimeError(
            f"Manifest remapping kept 0/{stats['total_rows']} rows under {data_root}."
        )

    rism_style_dir: Optional[Path] = (
        None if smoke_mode else (data_root / "unlabeled-10k-real" / "rism_hw")
    )
    return {
        "fullpage_manifest": dst_manifest,
        "rism_style_dir": rism_style_dir,
        "manifest_stats": stats,
    }


pull_hf_datasets = prepare_dataset


def restore_latest_from_hf(
    *,
    hf_token: str,
    out_dir: Path,
    latest_prefix: str = HF_LATEST_PREFIX,
    milestones_prefix: str = HF_MILESTONES_PREFIX,
) -> Dict[str, Any]:
    """Auto-resume from `checkpoints/iter15_pro200m_64k/latest/` on HF Private Hub."""
    from huggingface_hub import HfApi

    out_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi(token=hf_token)
    all_files = set(api.list_repo_files(repo_id=HF_REPO_ID, repo_type="dataset"))

    # Discover already-archived permanent milestones so we never re-upload them
    ms_re = re.compile(rf"^{re.escape(milestones_prefix.rstrip('/'))}/step_(\d+)_ema\.pt$")
    uploaded_milestones: Set[int] = set()
    for rf in all_files:
        m = ms_re.match(rf)
        if m:
            uploaded_milestones.add(int(m.group(1)))

    state: Dict[str, Any] = {
        "last_global_step": 0,
        "resume_ckpt": None,
        "uploaded_milestones": sorted(uploaded_milestones),
    }

    prefix = f"{latest_prefix.rstrip('/')}/"
    remote_latest = sorted(f for f in all_files if f.startswith(prefix))
    if not remote_latest:
        _log(f"No prior checkpoints under HF {prefix}; starting fresh from step 0.")
        return state

    _log(f"Found {len(remote_latest)} file(s) under HF {prefix}; restoring for auto-resume...")
    for fname in ROLLING_FILENAMES:
        rf = f"{prefix}{fname}"
        if rf not in all_files:
            continue
        dst = out_dir / fname
        _log(f"Restoring {rf} -> {dst}...")
        _download_hf_file(hf_token, rf, dst)

    state_file = out_dir / "run_state.json"
    saved_step = 0
    if state_file.is_file():
        try:
            loaded = json.loads(state_file.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                saved_step = int(loaded.get("last_global_step") or 0)
                for m_step in loaded.get("uploaded_milestones", []):
                    uploaded_milestones.add(int(m_step))
        except Exception as exc:
            _log(f"[WARN] Failed parsing restored run_state.json ({exc}); inferring from ckpt.")

    latest_ckpt = out_dir / "uni_omr_v2_latest.pt"
    ema_ckpt = out_dir / "uni_omr_v2_ema_latest.pt"
    resume_ckpt = latest_ckpt if latest_ckpt.is_file() else (ema_ckpt if ema_ckpt.is_file() else None)

    if resume_ckpt is not None:
        ckpt_step = _read_ckpt_step(resume_ckpt)
        last_global_step = max(saved_step, ckpt_step)
        state["last_global_step"] = last_global_step
        state["resume_ckpt"] = resume_ckpt
        _log(
            f"Auto-resume ready: --resume {resume_ckpt.name} "
            f"(ckpt_step={ckpt_step}, last_global_step={last_global_step})"
        )
    else:
        _log("No .pt file found under latest/; starting fresh from step 0.")

    state["uploaded_milestones"] = sorted(uploaded_milestones)
    return state


# ---------------------------------------------------------------------------
# 5. Exact `train_uni_v2.py` Command Builder
# ---------------------------------------------------------------------------


def build_iter15_train_cmd(
    *,
    python_exe: str,
    repo_root: Path,
    profile: GpuTuningProfile,
    fullpage_manifest: Path,
    rism_style_dir: Optional[Path],
    out_dir: Path,
    steps: int,
    warmup_steps: int = 500,
    resume_ckpt: Optional[Path] = None,
    init_encoder_from: Optional[Path] = None,
    teacher_ckpt: Optional[Path] = None,
    smoke_mode: bool = False,
) -> List[str]:
    """Construct the exact `train_uni_v2.py` CLI invocation for `iter15_pro200m`."""
    train_script = str(repo_root / "tmp-train" / "mvd" / "train_uni_v2.py")
    cmd: List[str] = [
        python_exe,
        "-u",
        train_script,
        "--preset",
        "iter15_pro200m",
        "--adaptive-ink-target",
        "--curriculum",
        "fullpage",
        "--fullpage-frac",
        "1.00",
        "--fullpage-manifest",
        str(fullpage_manifest),
        "--max-systems-per-page",
        "16",
        "--stratified-fullpage",
        "--stratified-preset",
        "v4_hw70",
        "--val-count",
        "4" if smoke_mode else "16",
        "--val-real-hw-count",
        "4" if smoke_mode else "16",
        "--fullpage-max-dim",
        "1536",
        "--multi-scale",
        "--fixed-hw-buckets",
        "--max-seq-len",
        "3840",
        "--batch-size",
        str(profile.batch_size),
        "--grad-accum",
        "1" if smoke_mode else str(profile.grad_accum),
        # Promoted optimizer & WSD schedule
        "--optimizer",
        "muon",
        "--muon-lr",
        "3.5e-4",
        "--lr",
        "2.0e-4",
        "--weight-decay",
        "0.01",
        "--no-decay-1d",
        "--schema-mask-v1",
        "--grad-clip",
        "1.0",
        "--lr-schedule",
        "wsd",
        "--warmup-steps",
        str(min(warmup_steps, max(1, steps // 4))) if smoke_mode else str(warmup_steps),
        "--wsd-decay-frac",
        "0.20",
        "--min-lr-ratio",
        "0.05",
        "--steps",
        str(steps),
        # Promoted losses, BD3 repair head, CASS, augmentation
        "--bd3-repair-weight",
        "0.25",
        "--bd3-n-views",
        "1",
        "--stroke-hog-weight",
        "0.15",
        "--state-token-loss-weight",
        "2.5",
        "--expr-token-loss-weight",
        "2.5",
        "--label-smoothing",
        "0.05",
        "--mixstyle-prob",
        "0.0",
        "--cass-batch-prob",
        "0.5",
        "--cass-prob",
        "0.30",
        "--cass-start-frac",
        "0.35",
        "--tok-dropout",
        "0.12",
        "--augment",
        "--aug-prob",
        "0.75",
        # Late-stage Hybrid INT4 QAT + GPU EMA + 5-min atomic rolling save + milestone saves
        "--qat-int4",
        "--qat-int4-start-frac",
        "0.85",
        "--qat-int4-block-size",
        "32",
        "--gpu-ema-decay",
        "0.99" if smoke_mode else "0.9985",
        "--gpu-ema-every",
        "1" if smoke_mode else "5",
        "--save-every-minutes",
        f"{SAVE_EVERY_MINUTES:.1f}",
        "--save-ema-milestones",
        "0",
        "--keep-ema-milestones",
        "0",
        "--best-min-step",
        "0" if smoke_mode else "4000",
        "--eval-every",
        str(max(1, steps)) if smoke_mode else "4000",
        "--eval-ema-only",
        "--eval-max-len",
        "256" if smoke_mode else "1024",
        "--log-every",
        "1" if smoke_mode else "10",
        "--num-workers",
        str(profile.num_workers),
        "--prefetch-threads",
        str(profile.prefetch_threads),
        "--seed",
        "42",
        "--out-dir",
        str(out_dir),
    ]

    if not smoke_mode:
        cmd.extend(
            [
                "--ema-milestone-steps",
                ",".join(str(s) for s in MILESTONE_STEPS),
            ]
        )

    if resume_ckpt is not None:
        cmd.extend(["--resume", str(resume_ckpt)])
    elif init_encoder_from is not None:
        cmd.extend(["--init-encoder-from", str(init_encoder_from)])

    if rism_style_dir is not None and rism_style_dir.is_dir():
        cmd.extend(
            [
                "--rism-style-dir",
                str(rism_style_dir),
                "--rism-fda-prob",
                "0.25",
                "--rism-adain-prob",
                "0.20",
                "--rism-fda-beta",
                "0.02",
            ]
        )

    if teacher_ckpt is not None and teacher_ckpt.is_file():
        cmd.extend(
            [
                "--teacher-ckpt",
                str(teacher_ckpt),
                "--teacher-max-dim",
                "1024",
                "--kd-beta",
                "0.5",
                "--kd-alpha",
                "0.30",
                "--kd-temp",
                "1.5",
                "--kd-teacher-ce-cap",
                "1.5",
                "--kd-hw-scale",
                "0.0",
                "--kd-prob",
                "0.5",
                "--kd-stop-frac",
                "0.65",
            ]
        )

    if profile.use_bf16:
        cmd.append("--bf16")
    if profile.use_grad_checkpoint:
        cmd.append("--grad-checkpoint")
    if profile.use_channels_last and sys.platform.startswith("linux"):
        cmd.append("--channels-last")
    if profile.use_no_cudnn:
        cmd.append("--no-cudnn")
    return cmd


# ---------------------------------------------------------------------------
# 6. CPU --dry-run Self-Test
# ---------------------------------------------------------------------------


def run_dry_run_self_test(repo_root: Path) -> None:
    _log("=== Running CPU --dry-run self-test ===")
    p_h100 = select_gpu_profile(vram_gb=80.0, device_name="NVIDIA H100 80GB")
    p_a100 = select_gpu_profile(vram_gb=40.0, device_name="NVIDIA A100-SXM4-40GB")
    p_l4 = select_gpu_profile(vram_gb=22.5, device_name="NVIDIA L4")
    assert p_h100.batch_size == 8 and p_h100.grad_accum == 2 and p_h100.num_workers == 8
    assert p_h100.batch_size * p_h100.grad_accum == 16
    assert p_a100.batch_size == 4 and p_a100.grad_accum == 4 and p_a100.num_workers == 6
    assert p_a100.batch_size * p_a100.grad_accum == 16
    assert p_l4.batch_size * p_l4.grad_accum == 16
    is_linux = sys.platform.startswith("linux")
    assert p_h100.use_channels_last is False
    assert p_a100.use_channels_last is False
    assert p_l4.use_channels_last is False
    assert os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") == "1"
    assert len(HF_ITER15_SHARD_FILES) == 14
    assert HF_ITER15_SHARD_FILES[0] == "datasets/iter15_v2a/shards/iter15_shard_00.tar"
    assert HF_ITER15_SHARD_FILES[-1] == "datasets/iter15_v2a/shards/iter15_shard_13.tar"
    assert HF_ITER15_MANIFEST_PATH == "datasets/iter15_v2a/shards/uni_fullpage_iter15_v2a.jsonl"

    with tempfile.TemporaryDirectory(prefix="iter15_pro200m_dryrun_") as td:
        tmp_root = Path(td)
        fake_data = tmp_root / "data"
        fake_out = tmp_root / "out"
        fake_rism = fake_data / "unlabeled-10k-real" / "rism_hw"
        fake_rism.mkdir(parents=True, exist_ok=True)
        fake_out.mkdir(parents=True, exist_ok=True)

        # Verify tar extraction helper + relative shard manifest remapping with drop_missing=True
        shard0_src = tmp_root / "shard0_src"
        (shard0_src / "pages" / "shard_00").mkdir(parents=True, exist_ok=True)
        (shard0_src / "bekerns" / "shard_00").mkdir(parents=True, exist_ok=True)
        (shard0_src / "pages" / "shard_00" / "p0.jpg").write_bytes(b"\xff\xd8\xff\xd9")
        (shard0_src / "bekerns" / "shard_00" / "p0_s0.bekern").write_text("**kern\n", encoding="utf-8")
        tar_path = tmp_root / "iter15_shard_00.tar"
        with tarfile.open(tar_path, "w") as tf:
            tf.add(shard0_src / "pages", arcname="pages")
            tf.add(shard0_src / "bekerns", arcname="bekerns")
        _extract_tar_to(tar_path, fake_data)
        assert (fake_data / "pages" / "shard_00" / "p0.jpg").is_file()

        src_jsonl = fake_data / "uni_fullpage_iter15_v2a.jsonl"
        dst_jsonl = fake_data / "uni_fullpage_iter15_v2a_colab.jsonl"
        src_jsonl.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "image": "pages/shard_00/p0.jpg",
                            "bekerns": ["bekerns/shard_00/p0_s0.bekern"],
                            "domain": "real_hw",
                        }
                    ),
                    json.dumps(
                        {
                            "image": "pages/shard_01/p1_missing.jpg",
                            "bekerns": ["bekerns/shard_01/p1_s0.bekern"],
                            "domain": "printed",
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        stats = remap_jsonl_manifest(src_jsonl, dst_jsonl, data_root=fake_data, drop_missing=True)
        assert stats["total_rows"] == 2 and stats["kept_rows"] == 1 and stats["missing_images"] == 1
        kept_rec = json.loads(dst_jsonl.read_text(encoding="utf-8").strip())
        assert kept_rec["image"] == (fake_data / "pages" / "shard_00" / "p0.jpg").as_posix()
        assert kept_rec["bekerns"] == [(fake_data / "bekerns" / "shard_00" / "p0_s0.bekern").as_posix()]

        # Verify full 64k CLI command (fresh & true --resume)
        fake_teacher = fake_data / "teacher_ema.pt"
        fake_teacher.write_bytes(b"teacher")
        fake_resume = fake_out / "uni_omr_v2_latest.pt"
        fake_resume.write_bytes(b"resume")

        cmd_full = build_iter15_train_cmd(
            python_exe=sys.executable,
            repo_root=repo_root,
            profile=p_h100,
            fullpage_manifest=dst_jsonl,
            rism_style_dir=fake_rism,
            out_dir=fake_out,
            steps=64000,
            resume_ckpt=fake_resume,
            teacher_ckpt=fake_teacher,
        )
        assert cmd_full[cmd_full.index("--preset") + 1] == "iter15_pro200m"
        assert cmd_full[cmd_full.index("--stratified-preset") + 1] == "v4_hw70"
        assert cmd_full[cmd_full.index("--bd3-repair-weight") + 1] == "0.25"
        assert cmd_full[cmd_full.index("--bd3-n-views") + 1] == "1"
        assert cmd_full[cmd_full.index("--fullpage-max-dim") + 1] == "1536"
        assert cmd_full[cmd_full.index("--steps") + 1] == "64000"
        assert cmd_full[cmd_full.index("--resume") + 1] == str(fake_resume)
        assert "--init-from" not in cmd_full
        assert cmd_full[cmd_full.index("--ema-milestone-steps") + 1] == "1000,2000,4000,8000,16000,32000,64000"
        assert cmd_full[cmd_full.index("--keep-ema-milestones") + 1] == "0"
        assert "--no-decay-1d" in cmd_full and "--schema-mask-v1" in cmd_full
        assert cmd_full[cmd_full.index("--save-every-minutes") + 1] == "5.0"
        assert cmd_full[cmd_full.index("--teacher-ckpt") + 1] == str(fake_teacher)
        assert "--qat-int4" in cmd_full and "--rism-fda-prob" in cmd_full
        assert ("--channels-last" in cmd_full) is is_linux

        # Verify smoke-steps CLI command (and missing rism_style_dir guard)
        missing_rism = fake_data / "nonexistent_rism_dir"
        cmd_smoke = build_iter15_train_cmd(
            python_exe=sys.executable,
            repo_root=repo_root,
            profile=p_h100,
            fullpage_manifest=dst_jsonl,
            rism_style_dir=missing_rism,
            out_dir=fake_out,
            steps=20,
            smoke_mode=True,
        )
        assert cmd_smoke[cmd_smoke.index("--steps") + 1] == "20"
        assert "--ema-milestone-steps" not in cmd_smoke
        assert "--rism-style-dir" not in cmd_smoke

        # Verify rolling + exact-only permanent milestone syncer (no wrong-step final-flush backfill!)
        uploaded: List[str] = []

        def _fake_upload(tmp_p: Path, repo_p: str, _msg: str) -> None:
            assert tmp_p.is_file() and tmp_p.name.endswith(".tmp")
            uploaded.append(repo_p)

        syncer = HFCheckpointAndMilestoneSyncer(
            out_dir=fake_out,
            initial_step=1000,
            already_uploaded_milestones=(1000,),
            interval_seconds=0.08,
            upload_fn=_fake_upload,
            ckpt_step_reader=lambda _p: 4500,  # > 4000, must NOT backfill step_004000_ema.pt!
        )
        syncer.start()
        try:
            (fake_out / "uni_omr_v2_latest.pt").write_bytes(b"latest-4.5k")
            (fake_out / "uni_omr_v2_ema_latest.pt").write_bytes(b"ema-4.5k")
            # Exact milestone file for step 2000 -> must upload!
            (fake_out / "uni_omr_v2_ema_step002000.pt").write_bytes(b"ema-ms-2k")
            time.sleep(0.25)
        finally:
            syncer.stop_and_final_sync()

        assert f"{HF_LATEST_PREFIX}/uni_omr_v2_latest.pt" in uploaded
        assert f"{HF_LATEST_PREFIX}/uni_omr_v2_ema_latest.pt" in uploaded
        assert f"{HF_MILESTONES_PREFIX}/step_002000_ema.pt" in uploaded
        assert f"{HF_MILESTONES_PREFIX}/step_001000_ema.pt" not in uploaded
        assert f"{HF_MILESTONES_PREFIX}/step_004000_ema.pt" not in uploaded
        assert 2000 in syncer.uploaded_milestones
        assert 4000 not in syncer.uploaded_milestones
        assert syncer.last_seen_global_step == 4500

        # Verify lightweight train_log.json step check avoids ckpt_step_reader
        ckpt_reader_calls = [0]

        def _counting_reader(_p: Path) -> int:
            ckpt_reader_calls[0] += 1
            return 9999

        (fake_out / "train_log.json").write_text(
            json.dumps([{"step": 100}, {"step": 5250}]) + "\n", encoding="utf-8"
        )
        syncer_log = HFCheckpointAndMilestoneSyncer(
            out_dir=fake_out,
            initial_step=0,
            dry_run=True,
            ckpt_step_reader=_counting_reader,
        )
        synced_n = syncer_log.sync_once(reason="dry-run-batch")
        assert synced_n >= 2
        assert syncer_log.last_seen_global_step == 5250
        assert ckpt_reader_calls[0] == 0

    _log("=== CPU --dry-run self-test PASSED ===")


# ---------------------------------------------------------------------------
# 7. CLI & Main Entrypoint
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Colab Recoverable Trainer for Iter 15 Flagship (`iter15_pro200m`, 64k)."
    )
    p.add_argument("--dry-run", action="store_true", help="Run CPU self-test and exit.")
    p.add_argument(
        "--smoke-steps",
        type=int,
        default=0,
        help="Fast pre-launch GPU verification for N steps (e.g. --smoke-steps 20; disables HF milestone archive).",
    )
    p.add_argument("--steps", type=int, default=TARGET_STEPS_DEFAULT, help="Total global steps (default: 64000).")
    p.add_argument("--warmup-steps", type=int, default=500, help="WSD warmup steps (default: 500).")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    p.add_argument("--manifest-name", type=str, default="uni_fullpage_iter15_v2a.jsonl")
    p.add_argument("--init-encoder-from", type=Path, default=None)
    p.add_argument("--teacher-ckpt", type=Path, default=None, help="Optional local teacher checkpoint override.")
    p.add_argument(
        "--teacher-hf-path",
        type=str,
        default=HF_TEACHER_CKPT_PATH,
        help="HF dataset path for KD teacher checkpoint (set empty string to disable).",
    )
    p.add_argument("--sync-interval-seconds", type=float, default=SYNC_INTERVAL_SECONDS)
    p.add_argument("--no-auto-resume", action="store_true", help="Ignore HF latest/ and start fresh.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = (
        os.environ.get("HF_HUB_ENABLE_HF_TRANSFER") or "1"
    )
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    repo_root = _resolve_repo_root()
    if args.dry_run:
        run_dry_run_self_test(repo_root)
        print(f"[bootstrap:{RUN_TAG}] Finished rc=0", flush=True)
        return 0

    _ensure_python_deps()
    hf_token = _read_hf_token(required=True)

    profile = select_gpu_profile()
    _log(
        f"Auto-tuned GPU profile: tier={profile.tier}, device={profile.device_name}, "
        f"vram={profile.vram_gb:.2f} GB, bs={profile.batch_size}, ga={profile.grad_accum}, "
        f"workers={profile.num_workers}, grad_ckpt={profile.use_grad_checkpoint}, "
        f"bf16={profile.use_bf16}, channels_last={profile.use_channels_last}"
    )

    smoke_mode = int(args.smoke_steps) > 0
    out_dir = (args.run_root / "smoke") if smoke_mode else args.run_root
    out_dir.mkdir(parents=True, exist_ok=True)

    assets = prepare_dataset(
        hf_token=hf_token,
        data_root=args.data_root,
        repo_root=repo_root,
        manifest_name=args.manifest_name,
        smoke_mode=smoke_mode,
    )

    teacher_ckpt: Optional[Path] = args.teacher_ckpt
    if teacher_ckpt is None and args.teacher_hf_path.strip():
        teacher_dst = args.data_root / "teachers" / "iter14_v2a_stage2_16k_ema_latest.pt"
        if not (teacher_dst.is_file() and teacher_dst.stat().st_size > 0):
            _log(f"Downloading KD teacher {args.teacher_hf_path} -> {teacher_dst}...")
            _download_hf_file(hf_token, args.teacher_hf_path.strip(), teacher_dst)
        teacher_ckpt = teacher_dst

    last_global_step = 0
    resume_ckpt: Optional[Path] = None
    uploaded_milestones: List[int] = []
    if not smoke_mode and not args.no_auto_resume:
        restored = restore_latest_from_hf(hf_token=hf_token, out_dir=out_dir)
        last_global_step = int(restored["last_global_step"])
        resume_ckpt = restored["resume_ckpt"]
        uploaded_milestones = list(restored["uploaded_milestones"])

    if smoke_mode:
        target_steps = int(args.smoke_steps)
        _log(f"=== SMOKE MODE: {target_steps} steps (no HF milestone archival) ===")
    else:
        target_steps = int(args.steps)
        if last_global_step >= target_steps:
            _log(
                f"Target global steps ({target_steps}) already reached "
                f"(last_global_step={last_global_step}). Nothing to do."
            )
            print(f"[bootstrap:{RUN_TAG}] Finished rc=0", flush=True)
            return 0
        _log(
            f"Launching Iter 15 pro200m: target_steps={target_steps} (fixed), "
            f"last_global_step={last_global_step}, resume_ckpt={resume_ckpt}, "
            f"uploaded_milestones={uploaded_milestones}"
        )

    syncer: Optional[HFCheckpointAndMilestoneSyncer] = None
    if not smoke_mode:
        syncer = HFCheckpointAndMilestoneSyncer(
            out_dir=out_dir,
            hf_repo_id=HF_REPO_ID,
            latest_prefix=HF_LATEST_PREFIX,
            milestones_prefix=HF_MILESTONES_PREFIX,
            hf_token=hf_token,
            initial_step=last_global_step,
            milestone_steps=MILESTONE_STEPS,
            already_uploaded_milestones=uploaded_milestones,
            archive_milestones=True,
            interval_seconds=float(args.sync_interval_seconds),
        )
        syncer.start()

    try:
        cmd = build_iter15_train_cmd(
            python_exe=sys.executable,
            repo_root=repo_root,
            profile=profile,
            fullpage_manifest=assets["fullpage_manifest"],
            rism_style_dir=assets["rism_style_dir"],
            out_dir=out_dir,
            steps=target_steps,
            warmup_steps=int(args.warmup_steps),
            resume_ckpt=resume_ckpt,
            init_encoder_from=args.init_encoder_from,
            teacher_ckpt=teacher_ckpt,
            smoke_mode=smoke_mode,
        )
        _run_cmd(cmd, cwd=repo_root)
        _log(f"=== {'SMOKE TEST' if smoke_mode else 'TRAINING'} COMPLETED CLEANLY ===")
        print(f"[bootstrap:{RUN_TAG}] Finished rc=0", flush=True)
        return 0
    except Exception:
        traceback.print_exc()
        print(f"[bootstrap:{RUN_TAG}] Finished rc=1", flush=True)
        return 1
    finally:
        if syncer is not None:
            syncer.stop_and_final_sync()
        _purge_tmp_token_file()


if __name__ == "__main__":
    raise SystemExit(main())
