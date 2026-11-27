#!/usr/bin/env python3
"""Google Colab H100 (with A100/L4 fallback) Recoverable 4-Phase Runner for Iter 15.

Experiment ID: 2026-11-27T07-40-iter15-edge85m-ax1gl5-vjepa-bekernv2a
Architecture:  UniOMR v2 edge85m (~81M params, < 55 MB WebGPU INT4) + ax0_gl6
Pipeline:
  - Phase 0: Stage-0 V-JEPA 2.1 + VICReg Pretraining (3,000 steps)
  - Phase 1: Stage-1 Strip Warmup (4,000 steps)
  - Phase 2: Stage-2 Full-Page (20,000 steps, bekern+v2 Phase A + Iter 14 KD + Late INT4 QAT)
  - Phase 3: Auto suite31_hw (Fast, +ILMS, INT4-dequant parity), 125-page OMR-NorthStar,
             and < 55 MB WebGPU INT4 ONNX export check + HF upload.

Features:
  - Auto-tunes batch size / grad accum / gradient checkpointing to H100-80GB (>=70 GB),
    A100-40GB (>=36 GB), or L4-24GB (<36 GB), using BF16 + cuDNN NCHW + fixed HxW buckets
    + BatchedMuon.
  - Bulletproof 5-minute recoverable HF checkpointing under
    `checkpoints/iter15-edge85m-h100/` via a non-blocking background syncer thread
    (mtime-gated, atomic .tmp copy, exponential backoff) + forced final sync in `finally`.
  - Zero-loss auto-resume on startup from HF `phase_state.json` + latest phase checkpoint.
  - Clean `D:\\tmp\\...` -> `/content/data/...` manifest path remapping.
  - Supports `--dry-run` on CPU to verify GPU auto-tuning, path remapping, command
    construction, and 5-minute sync helper without touching GPU or HF network.
"""

from __future__ import annotations

import argparse
import dataclasses
import fnmatch
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

EXP_ID = "2026-11-27T07-40-iter15-edge85m-ax1gl5-vjepa-bekernv2a"
RUN_TAG = "iter15-edge85m-h100"
HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_CKPT_PREFIX = f"checkpoints/{RUN_TAG}"
HF_ITER15_DATA_PREFIX = "datasets/iter15_v2a/"
HF_EVAL_PREFIX = f"evals/{EXP_ID}"
HF_RUN_LEDGER_PATH = f"runs/{EXP_ID}.json"

HF_TEACHER_CANDIDATES = (
    "checkpoints/iter14-v2a-final/uni_omr_v2_ema_latest.pt",
    "checkpoints/iter14-v2a-final/uni_omr_v2_ema_best.pt",
    "checkpoints/iter13_v2_mid/uni_omr_v2_ema_latest.pt",
    "checkpoints/iter12_v2_best/uni_omr_v2_ema_best.pt",
)

HF_CORE_DATASET_SHARDS = (
    ("datasets/rism_hw_5707.tar", "unlabeled-10k-real/rism_hw"),
    ("datasets/hw_sim_v06_500.tar", "hw-sim-v06-historical-clean"),
    ("datasets/suite31_hw_eval.tar", "suite31_hw_eval"),
    ("datasets/northstar_eval_pack_125p.tar.gz", "northstar-eval-pack"),
)

BUNDLE_CANDIDATES = (
    Path("/content/iter15_bundle.tar.gz"),
    Path("/content/ov_bundle.tar.gz"),
    Path("/content/bundle.tar.gz"),
    Path("/content/repo_bundle.tar.gz"),
)

DEFAULT_DATA_ROOT = Path("/content/data")
DEFAULT_RUN_ROOT = Path("/content/runs/iter15-edge85m-h100")
HF_TOKEN_FILE = Path("/tmp/.hf_token")

SYNC_PATTERNS = (
    "uni_omr_v2_vjepa_autosave.pt",
    "uni_omr_v2_vjepa_best.pt",
    "uni_omr_v2_vjepa_last.pt",
    "uni_omr_v2_latest.pt",
    "uni_omr_v2_ema_latest.pt",
    "uni_omr_v2_ema_best.pt",
    "uni_omr_v2_ema_step*.pt",
    "train_log.json",
    "phase_state.json",
)

WEBGPU_MAX_MB = 55.0
SAVE_EVERY_MINUTES = 5.0
STAGE0_STEPS_DEFAULT = 1500
STAGE1_STEPS_DEFAULT = 2000
STAGE2_STEPS_DEFAULT = 12000
STAGE2_WARMUP_DEFAULT = 300
STAGE2_PROG_RES_SCHEDULE = "0.0:896,0.35:1152,0.65:1408"
STAGE2_CASS_START_FRAC = 0.35


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
# 1. GPU Tier Auto-Tuning (H100-80GB / A100-40GB / L4-24GB)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class GpuTuningProfile:
    tier: str
    device_name: str
    vram_gb: float
    batch_size: int
    grad_accum: int
    strip_batch_size: int
    strip_grad_accum: int
    use_grad_checkpoint: bool
    use_bf16: bool = True
    use_no_cudnn: bool = False  # Linux Colab cuDNN NCHW + fixed HxW buckets is fast
    prefetch_threads: int = 6
    num_workers: int = 0


def select_gpu_profile(
    *,
    vram_gb: Optional[float] = None,
    device_name: Optional[str] = None,
) -> GpuTuningProfile:
    """Auto-tune training knobs for H100 (>=70 GB), A100 (>=36 GB), or L4 (<36 GB)."""
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

    dev_name = device_name or f"GPU-{vram_gb:.1f}GB"
    if vram_gb >= 70.0:
        return GpuTuningProfile(
            tier="H100-80GB",
            device_name=dev_name,
            vram_gb=round(vram_gb, 2),
            batch_size=16,
            grad_accum=1,
            strip_batch_size=64,
            strip_grad_accum=1,
            use_grad_checkpoint=False,
            use_bf16=True,
            use_no_cudnn=False,
            prefetch_threads=8,
            num_workers=8,
        )
    if vram_gb >= 36.0:
        return GpuTuningProfile(
            tier="A100-40GB",
            device_name=dev_name,
            vram_gb=round(vram_gb, 2),
            batch_size=4,
            grad_accum=2,
            strip_batch_size=8,
            strip_grad_accum=4,
            use_grad_checkpoint=True,
            use_bf16=True,
            use_no_cudnn=False,
            prefetch_threads=6,
            num_workers=4,
        )
    return GpuTuningProfile(
        tier="L4-24GB",
        device_name=dev_name,
        vram_gb=round(vram_gb, 2),
        batch_size=2,
        grad_accum=4,
        strip_batch_size=8,
        strip_grad_accum=4,
        use_grad_checkpoint=True,
        use_bf16=True,
        use_no_cudnn=False,
        prefetch_threads=4,
        num_workers=4,
    )


# ---------------------------------------------------------------------------
# 2. Clean Manifest Path Remapper (D:\tmp\... -> /content/data/...)
# ---------------------------------------------------------------------------

_WIN_DRIVE_TMP_RE = re.compile(r"^[A-Za-z]:[/\\]+tmp[/\\]+", re.IGNORECASE)
_WIN_REPO_RE = re.compile(
    r"^[A-Za-z]:[/\\]+.*?sheet-music-converter[/\\]+", re.IGNORECASE
)


def remap_windows_path(
    raw_path: str,
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    repo_root: Optional[Path] = None,
) -> str:
    """Remap Windows `D:\\tmp\\...` (or repo-local) paths cleanly onto Linux `data_root`."""
    if not raw_path:
        return raw_path
    s = raw_path.strip()
    if not s:
        return s

    # Normalize backslashes to POSIX separators for consistent prefix matching
    norm = s.replace("\\", "/")

    # 1. Strip `D:/tmp/` (or any `<drive>:/tmp/`) -> `<data_root>/<rel>`
    m_tmp = _WIN_DRIVE_TMP_RE.match(s)
    if m_tmp:
        rel = norm[m_tmp.end() :].lstrip("/")
        return (data_root / rel).as_posix()

    # 2. Strip Windows repo prefix `C:/.../sheet-music-converter/...` -> `<repo_root>/<rel>`
    m_repo = _WIN_REPO_RE.match(s)
    if m_repo:
        rel = norm[m_repo.end() :].lstrip("/")
        base = repo_root if repo_root is not None else data_root
        return (base / rel).as_posix()

    # 3. Already under data_root or absolute POSIX path
    if norm.startswith("/"):
        return norm

    # 4. Relative path: if it starts with `tmp-train/` and repo_root is known, resolve against repo_root
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
    """Remap all path fields in a JSONL manifest from `D:\\tmp\\...` to `data_root`."""
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

        for key in ("image", "img", "image_path", "musicxml", "src_path"):
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


# ---------------------------------------------------------------------------
# 3. Bulletproof 5-Minute Background HF Checkpoint Syncer
# ---------------------------------------------------------------------------


def _matches_sync_pattern(filename: str, patterns: Sequence[str] = SYNC_PATTERNS) -> bool:
    return any(fnmatch.fnmatch(filename, pat) for pat in patterns)


class HFCheckpointSyncer:
    """Non-blocking background thread syncing modified checkpoints to HF every 5 min.

    - Polls registered `(subdir_tag, local_dir)` roots + root `phase_state.json`.
    - Tracks `(rel_repo_path) -> (mtime_ns, size)` so unchanged files are never re-uploaded.
    - Copies each target file atomically to a `.sync.tmp` snapshot before upload so an
      in-progress local save cannot corrupt the uploaded bytes.
    - Retries transient HF network errors with exponential backoff.
    - Supports `dry_run=True` or custom `upload_fn` for fast CPU unit verification.
    """

    def __init__(
        self,
        *,
        run_root: Path,
        hf_repo_id: str = HF_REPO_ID,
        hf_ckpt_prefix: str = HF_CKPT_PREFIX,
        hf_token: str = "",
        interval_seconds: float = 300.0,
        max_retries: int = 5,
        base_backoff_s: float = 2.0,
        dry_run: bool = False,
        upload_fn: Optional[Callable[[Path, str, str], None]] = None,
    ) -> None:
        self.run_root = run_root
        self.hf_repo_id = hf_repo_id
        self.hf_ckpt_prefix = hf_ckpt_prefix.rstrip("/")
        self.hf_token = hf_token
        self.interval_seconds = max(0.05, float(interval_seconds))
        self.max_retries = max(1, int(max_retries))
        self.base_backoff_s = max(0.01, float(base_backoff_s))
        self.dry_run = dry_run
        self._upload_fn = upload_fn

        self._watch_dirs: Dict[str, Path] = {}
        self._synced_meta: Dict[str, Tuple[int, int]] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.upload_history: List[Dict[str, Any]] = []
        self.error_count = 0

    def register_dir(self, repo_subdir: str, local_dir: Path) -> None:
        with self._lock:
            self._watch_dirs[repo_subdir.strip("/")] = local_dir

    def _collect_candidates(self) -> List[Tuple[Path, str]]:
        candidates: List[Tuple[Path, str]] = []
        # Always include root phase_state.json if present
        state_file = self.run_root / "phase_state.json"
        if state_file.is_file():
            candidates.append((state_file, f"{self.hf_ckpt_prefix}/phase_state.json"))

        with self._lock:
            watch_items = list(self._watch_dirs.items())

        for subdir, local_dir in watch_items:
            if not local_dir.is_dir():
                continue
            for entry in sorted(local_dir.iterdir()):
                if not entry.is_file():
                    continue
                if entry.name.endswith(".tmp"):
                    continue
                if _matches_sync_pattern(entry.name):
                    rel_repo = (
                        f"{self.hf_ckpt_prefix}/{subdir}/{entry.name}"
                        if subdir
                        else f"{self.hf_ckpt_prefix}/{entry.name}"
                    )
                    candidates.append((entry, rel_repo))
        return candidates

    def _upload_with_backoff(self, src_path: Path, path_in_repo: str, commit_msg: str) -> bool:
        # Stage an atomic snapshot copy so concurrent local saves cannot race the upload
        stage_dir = self.run_root / ".sync_stage"
        stage_dir.mkdir(parents=True, exist_ok=True)
        tmp_snap = stage_dir / f"{src_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        try:
            shutil.copy2(src_path, tmp_snap)
            size_bytes = tmp_snap.stat().st_size
            for attempt in range(1, self.max_retries + 1):
                try:
                    if self._upload_fn is not None:
                        self._upload_fn(tmp_snap, path_in_repo, commit_msg)
                    elif self.dry_run:
                        _log(f"[HF-Sync DRY-RUN] would upload {src_path.name} ({size_bytes} B) -> {path_in_repo}")
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

    def sync_once(self, *, reason: str = "periodic") -> int:
        """Upload any tracked files whose (mtime_ns, size) changed since last sync."""
        uploaded = 0
        for local_path, path_in_repo in self._collect_candidates():
            try:
                st = local_path.stat()
            except OSError:
                continue
            sig = (int(st.st_mtime_ns), int(st.st_size))
            if self._synced_meta.get(path_in_repo) == sig:
                continue
            commit_msg = f"ckpt({RUN_TAG}): {reason} sync {local_path.name}"
            if self._upload_with_backoff(local_path, path_in_repo, commit_msg):
                self._synced_meta[path_in_repo] = sig
                uploaded += 1
        if uploaded > 0:
            _log(f"[HF-Sync] Synced {uploaded} updated file(s) ({reason}) -> {self.hf_ckpt_prefix}/")
        return uploaded

    def _run_loop(self) -> None:
        _log(
            f"[HF-Sync] Background 5-min sync thread started "
            f"(interval={self.interval_seconds:.1f}s, prefix={self.hf_ckpt_prefix}/)"
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
            target=self._run_loop, name="hf-ckpt-syncer", daemon=True
        )
        self._thread.start()

    def stop_and_final_sync(self) -> int:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=30.0)
        return self.sync_once(reason="final-flush")


# ---------------------------------------------------------------------------
# 4. Environment, Repo, Rust Binaries, and Subprocess Helpers
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
            extract_root = Path("/workspace")
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


def _ensure_python_deps(*, include_onnx: bool = True) -> None:
    pkgs: List[str] = []
    checks: List[Tuple[str, str]] = [
        ("huggingface_hub", "huggingface_hub>=0.26"),
        ("PIL", "pillow"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("tqdm", "tqdm"),
    ]
    if include_onnx:
        checks.extend(
            [
                ("onnx", "onnx"),
                ("onnxruntime", "onnxruntime"),
                ("onnx_ir", "onnx-ir"),
                ("onnxscript", "onnxscript"),
            ]
        )
    for mod, pkg in checks:
        try:
            __import__(mod)
        except ImportError:
            pkgs.append(pkg)
    if pkgs:
        _run_cmd([sys.executable, "-m", "pip", "install", "-q", *pkgs])
        importlib.invalidate_caches()


def _ensure_rust_eval_binaries(repo_root: Path) -> Tuple[Path, Path]:
    ext = ".exe" if os.name == "nt" else ""
    aser_manifest = repo_root / "converter" / "assembly-serialize" / "Cargo.toml"
    eharness_manifest = repo_root / "converter" / "eval-harness" / "Cargo.toml"
    aser_bin = aser_manifest.parent / "target" / "release" / f"assembly-serialize{ext}"
    eharness_bin = eharness_manifest.parent / "target" / "release" / f"eval-harness{ext}"

    if aser_bin.is_file() and eharness_bin.is_file():
        return aser_bin, eharness_bin

    cargo_bin_dir = Path.home() / ".cargo" / "bin"
    env = os.environ.copy()
    sep = ";" if os.name == "nt" else ":"
    env["PATH"] = f"{cargo_bin_dir}{sep}{env.get('PATH', '')}"
    if shutil.which("cargo", path=env["PATH"]) is None:
        if os.name == "nt":
            raise RuntimeError("cargo not found on Windows PATH")
        _log("Installing minimal Rust toolchain via rustup...")
        _run_cmd(
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
            "| sh -s -- -y --default-toolchain stable --profile minimal",
            shell=True,
            env=env,
        )
    if not aser_bin.is_file():
        _run_cmd(
            ["cargo", "build", "--release", "--manifest-path", str(aser_manifest)],
            env=env,
        )
    if not eharness_bin.is_file():
        _run_cmd(
            ["cargo", "build", "--release", "--manifest-path", str(eharness_manifest)],
            env=env,
        )
    return aser_bin, eharness_bin


# ---------------------------------------------------------------------------
# 5. HF Dataset Pull, Teacher Pull, and Zero-Loss Auto-Resume
# ---------------------------------------------------------------------------


def _extract_tar_to(tar_path: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as tf:
        tf.extractall(path=dst_dir)


def _download_hf_file(
    hf_token: str,
    repo_filename: str,
    dst_path: Path,
) -> Path:
    from huggingface_hub import hf_hub_download

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dl = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=repo_filename,
        repo_type="dataset",
        token=hf_token,
        local_dir="/tmp/hf_dl_cache",
    )
    shutil.copy2(dl, dst_path)
    try:
        os.remove(dl)
    except OSError:
        pass
    return dst_path


def pull_datasets_and_teacher(
    *,
    hf_token: str,
    data_root: Path,
    repo_root: Path,
) -> Dict[str, Any]:
    """Pull HF dataset shards (`datasets/iter15_v2a/*` + core shards) and Iter 14 teacher."""
    from huggingface_hub import HfApi

    data_root.mkdir(parents=True, exist_ok=True)
    api = HfApi(token=hf_token)
    all_files = set(api.list_repo_files(repo_id=HF_REPO_ID, repo_type="dataset"))

    # 1. Pull everything under `datasets/iter15_v2a/*`
    iter15_files = sorted(f for f in all_files if f.startswith(HF_ITER15_DATA_PREFIX))
    _log(f"Found {len(iter15_files)} file(s) under HF {HF_ITER15_DATA_PREFIX}")
    for rf in iter15_files:
        rel_name = rf[len(HF_ITER15_DATA_PREFIX) :]
        marker = data_root / ".unpacked" / f"iter15_v2a__{rel_name.replace('/', '__')}.done"
        if marker.is_file():
            continue
        tmp_dst = data_root / ".downloads" / rel_name
        _log(f"Downloading {rf}...")
        _download_hf_file(hf_token, rf, tmp_dst)
        if rel_name.endswith((".tar", ".tar.gz", ".tgz")):
            _log(f"Extracting {rel_name} -> {data_root}...")
            _extract_tar_to(tmp_dst, data_root)
            try:
                tmp_dst.unlink()
            except OSError:
                pass
        else:
            # Manifest or loose metadata file placed directly under data_root
            target = data_root / Path(rel_name).name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(tmp_dst), str(target))
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(_utc_now() + "\n", encoding="utf-8")

    # 2. Pull core shared shards (RISM HW, hw_sim_v06, suite31_hw, NorthStar 125p) if absent
    for rf, target_subdir in HF_CORE_DATASET_SHARDS:
        if rf not in all_files:
            continue
        marker = data_root / ".unpacked" / f"{Path(rf).name}.done"
        if marker.is_file():
            continue
        dst_dir = data_root / target_subdir
        tmp_dst = data_root / ".downloads" / Path(rf).name
        _log(f"Downloading core shard {rf} -> {dst_dir}...")
        _download_hf_file(hf_token, rf, tmp_dst)
        _extract_tar_to(tmp_dst, dst_dir)
        try:
            tmp_dst.unlink()
        except OSError:
            pass
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(_utc_now() + "\n", encoding="utf-8")

    # 3. Remap any JSONL manifests (iter14 printed manifest + iter15 Stage-2 manifest)
    remapped_manifests: Dict[str, Path] = {}
    for name in (
        "uni_fullpage_iter15_v2a.jsonl",
        "uni_fullpage_iter14_v2a.jsonl",
    ):
        src = data_root / name
        if src.is_file():
            dst = data_root / f"{src.stem}_colab.jsonl"
            stats = remap_jsonl_manifest(
                src, dst, data_root=data_root, repo_root=repo_root, drop_missing=False
            )
            _log(f"Remapped manifest {name} -> {dst.name}: {stats}")
            remapped_manifests[name] = dst

    # 4. Pull Iter 14 265-vocab EMA teacher checkpoint (with fallback to iter13/iter12)
    teacher_dir = data_root / "teachers"
    teacher_dir.mkdir(parents=True, exist_ok=True)
    teacher_local: Optional[Path] = None
    teacher_source: Optional[str] = None
    for cand in HF_TEACHER_CANDIDATES:
        if cand not in all_files:
            continue
        target = teacher_dir / cand.replace("/", "__")
        if not target.is_file():
            _log(f"Downloading teacher checkpoint {cand}...")
            _download_hf_file(hf_token, cand, target)
        teacher_local = target
        teacher_source = cand
        break
    if teacher_local is None:
        raise RuntimeError(
            f"No teacher checkpoint found on HF among {HF_TEACHER_CANDIDATES}"
        )
    _log(f"Selected teacher checkpoint: {teacher_source} -> {teacher_local}")

    # Resolve key data directories
    rism_dir_a = data_root / "unlabeled-10k-real" / "rism_hw"
    rism_dir_b = data_root / "rism-pdm-hw" / "pages"
    printed_strip = repo_root / "tmp-train" / "mvd-sysdata-t32"
    if not printed_strip.is_dir() and (data_root / "mvd-sysdata-t32").is_dir():
        printed_strip = data_root / "mvd-sysdata-t32"
    hw_strip = data_root / "real-hw-strips-iter7"
    hw_strip_v06 = data_root / "hw-sim-v06-historical-clean" / "stage2_strips"
    if not hw_strip_v06.is_dir():
        hw_strip_v06 = data_root / "hw-sim-v06-historical-clean"

    suite31_manifest = data_root / "suite31_hw_eval" / "suite31" / "suite31_hw_manifest.json"
    northstar_manifest = data_root / "northstar-eval-pack" / "manifest.json"
    if not northstar_manifest.is_file():
        nested_ns = data_root / "northstar-eval-pack" / "northstar-eval-pack" / "manifest.json"
        if nested_ns.is_file():
            northstar_manifest = nested_ns

    return {
        "rism_dir_a": rism_dir_a,
        "rism_dir_b": rism_dir_b,
        "printed_manifest": remapped_manifests.get(
            "uni_fullpage_iter14_v2a.jsonl",
            remapped_manifests.get("uni_fullpage_iter15_v2a.jsonl"),
        ),
        "stage2_manifest": remapped_manifests.get("uni_fullpage_iter15_v2a.jsonl"),
        "printed_strip": printed_strip,
        "hw_strip": hw_strip,
        "hw_strip_v06": hw_strip_v06,
        "teacher_ckpt": teacher_local,
        "teacher_source": teacher_source,
        "suite31_manifest": suite31_manifest,
        "northstar_manifest": northstar_manifest,
    }


def restore_from_hf_if_present(
    *,
    hf_token: str,
    run_root: Path,
    phase_dirs: Dict[str, Path],
) -> Dict[str, Any]:
    """Check HF `checkpoints/iter15-edge85m-h100/` and restore phase_state + latest ckpts."""
    from huggingface_hub import HfApi

    default_state: Dict[str, Any] = {
        "exp_id": EXP_ID,
        "run_tag": RUN_TAG,
        "active_phase": 0,
        "completed_phases": [],
        "updated_utc": _utc_now(),
    }

    api = HfApi(token=hf_token)
    all_files = set(api.list_repo_files(repo_id=HF_REPO_ID, repo_type="dataset"))
    prefix = f"{HF_CKPT_PREFIX}/"
    remote_ckpts = sorted(f for f in all_files if f.startswith(prefix))
    if not remote_ckpts:
        _log(f"No prior checkpoints found under HF {prefix}; starting fresh at Phase 0.")
        return default_state

    _log(f"Found {len(remote_ckpts)} remote artifact(s) under HF {prefix}; restoring...")
    state_repo_path = f"{HF_CKPT_PREFIX}/phase_state.json"
    state = dict(default_state)
    if state_repo_path in all_files:
        local_state = run_root / "phase_state.json"
        _download_hf_file(hf_token, state_repo_path, local_state)
        try:
            loaded = json.loads(local_state.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                state.update(loaded)
        except Exception as exc:
            _log(f"[WARN] Could not parse remote phase_state.json ({exc}); inferring state.")

    # Download relevant checkpoints for each phase directory
    for phase_key, local_dir in phase_dirs.items():
        local_dir.mkdir(parents=True, exist_ok=True)
        sub_prefix = f"{HF_CKPT_PREFIX}/{phase_key}/"
        for rf in remote_ckpts:
            if not rf.startswith(sub_prefix):
                continue
            fname = rf[len(sub_prefix) :]
            if "/" in fname:
                continue
            # Always restore latest/best/autosave/last/log (skip huge old milestones unless needed)
            if fname in (
                "uni_omr_v2_vjepa_autosave.pt",
                "uni_omr_v2_vjepa_best.pt",
                "uni_omr_v2_vjepa_last.pt",
                "uni_omr_v2_latest.pt",
                "uni_omr_v2_ema_latest.pt",
                "uni_omr_v2_ema_best.pt",
                "train_log.json",
            ):
                dst = local_dir / fname
                if not dst.is_file():
                    _log(f"Restoring {rf} -> {dst}...")
                    _download_hf_file(hf_token, rf, dst)

    _log(
        f"Auto-resume state: active_phase={state.get('active_phase')}, "
        f"completed_phases={state.get('completed_phases')}"
    )
    return state


def _read_ckpt_step(ckpt_path: Path) -> int:
    try:
        import torch

        ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        if isinstance(ck, dict):
            return int(ck.get("step", 0) or 0)
    except Exception:
        pass
    return 0


# ---------------------------------------------------------------------------
# 6. Command Builders for Phases 0, 1, 2 (exact parity with launch_iter15.ps1)
# ---------------------------------------------------------------------------


def build_phase0_cmd(
    *,
    python_exe: str,
    repo_root: Path,
    profile: GpuTuningProfile,
    rism_dirs: Sequence[Path],
    printed_manifest: Optional[Path],
    out_dir: Path,
    steps: int = STAGE0_STEPS_DEFAULT,
    resume_ckpt: Optional[Path] = None,
) -> List[str]:
    vjepa_script = str(repo_root / "tmp-train" / "mvd" / "pretrain_vjepa_iter15.py")
    existing_rism = [str(d) for d in rism_dirs if d.is_dir()]
    if not existing_rism:
        existing_rism = [str(rism_dirs[0])]

    cmd = [
        python_exe,
        "-u",
        vjepa_script,
        "--arch-preset",
        "edge85m",
        "--n-axial-layers",
        "0",
        "--n-global-layers",
        "6",
        "--max-mem-tokens",
        "3072",
        "--adaptive-ink-target",
        "--image-dirs",
        *existing_rism,
    ]
    if printed_manifest is not None:
        cmd.extend(["--printed-manifest", str(printed_manifest), "--printed-max-pages", "6000"])

    effective_steps = steps
    if resume_ckpt is not None and resume_ckpt.is_file():
        done_steps = _read_ckpt_step(resume_ckpt)
        effective_steps = max(1, steps - done_steps)
        cmd.extend(["--init-from", str(resume_ckpt), "--gram-weight", "0.0"])

    cmd.extend(
        [
            "--mask-ratio",
            "0.45",
            "--mask-half-width",
            "0.05",
            "--vicreg-var-weight",
            "1.0",
            "--vicreg-cov-weight",
            "0.04",
            "--disable-random-gram",
            "--stroke-hog-weight",
            "0.25",
            "--router-cons-weight",
            "0.2",
            "--optimizer",
            "muon",
            "--lr",
            "3.0e-4",
            "--weight-decay",
            "0.04",
            "--warmup-steps",
            "150",
            "--steps",
            str(effective_steps),
            "--batch-size",
            str(profile.batch_size),
            "--grad-accum",
            str(profile.grad_accum),
            "--max-dim",
            "1024",
            "--prefetch-threads",
            str(profile.prefetch_threads),
            "--val-probe-pages",
            "128",
            "--save-every",
            "500",
            "--save-every-minutes",
            f"{SAVE_EVERY_MINUTES:.1f}",
            "--seed",
            "42",
            "--out-dir",
            str(out_dir),
        ]
    )
    if profile.use_bf16:
        cmd.append("--bf16")
    if profile.use_grad_checkpoint:
        cmd.append("--grad-checkpoint")
    else:
        cmd.append("--no-grad-checkpoint")
    if profile.use_no_cudnn:
        cmd.append("--no-cudnn")
    return cmd


def build_phase1_cmd(
    *,
    python_exe: str,
    repo_root: Path,
    profile: GpuTuningProfile,
    phase0_ckpt: Path,
    printed_strip: Path,
    hw_strips: Sequence[Path],
    out_dir: Path,
    steps: int = STAGE1_STEPS_DEFAULT,
    resume_ckpt: Optional[Path] = None,
) -> List[str]:
    train_script = str(repo_root / "tmp-train" / "mvd" / "train_uni_v2.py")
    existing_hw = [str(d) for d in hw_strips if d.is_dir()]
    if not existing_hw:
        existing_hw = [str(hw_strips[0])]

    cmd = [
        python_exe,
        "-u",
        train_script,
        "--preset",
        "edge85m",
        "--n-axial-layers",
        "0",
        "--n-global-layers",
        "6",
        "--max-mem-tokens",
        "3072",
        "--adaptive-ink-target",
    ]
    if resume_ckpt is not None and resume_ckpt.is_file():
        cmd.extend(["--resume", str(resume_ckpt)])
    else:
        cmd.extend(["--init-encoder-from", str(phase0_ckpt)])

    cmd.extend(
        [
            "--curriculum",
            "strip",
            "--strip-dirs",
            str(printed_strip),
            "--hw-strip-dirs",
            *existing_hw,
            "--hw-frac",
            "0.30",
            "--strip-batch-size",
            str(profile.strip_batch_size),
            "--strip-grad-accum",
            str(profile.strip_grad_accum),
            "--grad-accum",
            str(profile.strip_grad_accum),
            "--strip-height",
            "192",
            "--max-strip-width",
            "1024",
            "--max-seq-len",
            "1024",
            "--optimizer",
            "muon",
            "--muon-lr",
            "8.0e-4",
            "--lr",
            "2.5e-4",
            "--weight-decay",
            "0.01",
            "--no-decay-1d",
            "--grad-clip",
            "1.0",
            "--lr-schedule",
            "wsd",
            "--warmup-steps",
            "200",
            "--wsd-decay-frac",
            "0.15",
            "--min-lr-ratio",
            "0.05",
            "--steps",
            str(steps),
            "--mixstyle-prob",
            "0.0",
            "--cass-prob",
            "0.15",
            "--tok-dropout",
            "0.12",
            "--augment",
            "--aug-prob",
            "0.75",
            "--state-token-loss-weight",
            "2.5",
            "--expr-token-loss-weight",
            "2.5",
            "--gpu-ema-decay",
            "0.995",
            "--gpu-ema-every",
            "2",
            "--eval-every",
            "2000",
            "--eval-ema-only",
            "--val-count",
            "32",
            "--save-every-minutes",
            f"{SAVE_EVERY_MINUTES:.1f}",
            "--num-workers",
            str(profile.num_workers),
            "--prefetch-threads",
            str(profile.prefetch_threads),
            "--seed",
            "42",
            "--out-dir",
            str(out_dir),
        ]
    )
    if profile.use_bf16:
        cmd.append("--bf16")
    # train_uni_v2.py uses store_true `--grad-checkpoint`; omit on H100 (>=70 GB)
    if profile.use_grad_checkpoint:
        cmd.append("--grad-checkpoint")
    if profile.use_no_cudnn:
        cmd.append("--no-cudnn")
    return cmd


def build_phase2_cmd(
    *,
    python_exe: str,
    repo_root: Path,
    profile: GpuTuningProfile,
    phase1_ckpt: Path,
    stage2_manifest: Path,
    rism_style_dir: Path,
    teacher_ckpt: Path,
    out_dir: Path,
    steps: int = STAGE2_STEPS_DEFAULT,
    warmup_steps: int = STAGE2_WARMUP_DEFAULT,
    resume_ckpt: Optional[Path] = None,
) -> List[str]:
    train_script = str(repo_root / "tmp-train" / "mvd" / "train_uni_v2.py")
    cmd = [
        python_exe,
        "-u",
        train_script,
        "--preset",
        "edge85m",
        "--n-axial-layers",
        "0",
        "--n-global-layers",
        "6",
        "--max-mem-tokens",
        "3072",
        "--adaptive-ink-target",
    ]
    if resume_ckpt is not None and resume_ckpt.is_file():
        cmd.extend(["--resume", str(resume_ckpt)])
    else:
        cmd.extend(["--init-from", str(phase1_ckpt)])

    cmd.extend(
        [
            "--curriculum",
            "fullpage",
            "--fullpage-frac",
            "1.00",
            "--fullpage-manifest",
            str(stage2_manifest),
            "--max-systems-per-page",
            "16",
            "--stratified-fullpage",
            "--stratified-preset",
            "v3",
            "--val-count",
            "16",
            "--val-real-hw-count",
            "16",
            "--fullpage-max-dim",
            "1408",
            "--prog-res-schedule",
            STAGE2_PROG_RES_SCHEDULE,
            "--multi-scale",
            "--fixed-hw-buckets",
            "--max-seq-len",
            "3840",
            "--batch-size",
            str(profile.batch_size),
            "--grad-accum",
            str(profile.grad_accum),
            "--optimizer",
            "muon",
            "--muon-lr",
            "3.5e-4",
            "--lr",
            "2.0e-4",
            "--weight-decay",
            "0.01",
            "--no-decay-1d",
            "--grad-clip",
            "1.0",
            "--lr-schedule",
            "wsd",
            "--warmup-steps",
            str(warmup_steps),
            "--wsd-decay-frac",
            "0.20",
            "--min-lr-ratio",
            "0.05",
            "--steps",
            str(steps),
            "--mixstyle-prob",
            "0.0",
            "--cass-batch-prob",
            "0.5",
            "--cass-prob",
            "0.30",
            "--cass-start-frac",
            f"{STAGE2_CASS_START_FRAC:.2f}",
            "--tok-dropout",
            "0.12",
            "--augment",
            "--aug-prob",
            "0.75",
            "--rism-style-dir",
            str(rism_style_dir),
            "--rism-fda-prob",
            "0.25",
            "--rism-adain-prob",
            "0.20",
            "--rism-fda-beta",
            "0.02",
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
            "--state-token-loss-weight",
            "2.5",
            "--expr-token-loss-weight",
            "2.5",
            "--qat-int4",
            "--qat-int4-start-frac",
            "0.85",
            "--qat-int4-block-size",
            "32",
            "--gpu-ema-decay",
            "0.9985",
            "--gpu-ema-every",
            "5",
            "--save-ema-milestones",
            "4000",
            "--best-min-step",
            "4000",
            "--eval-every",
            "4000",
            "--eval-ema-only",
            "--eval-max-len",
            "1024",
            "--save-every-minutes",
            f"{SAVE_EVERY_MINUTES:.1f}",
            "--num-workers",
            str(profile.num_workers),
            "--prefetch-threads",
            str(profile.prefetch_threads),
            "--seed",
            "42",
            "--out-dir",
            str(out_dir),
        ]
    )
    if profile.use_bf16:
        cmd.append("--bf16")
    # train_uni_v2.py uses store_true `--grad-checkpoint`; omit on H100 (>=70 GB)
    if profile.use_grad_checkpoint:
        cmd.append("--grad-checkpoint")
    if profile.use_no_cudnn:
        cmd.append("--no-cudnn")
    return cmd


# ---------------------------------------------------------------------------
# 7. Phase 3: Auto Evals (suite31_hw Fast, +ILMS, INT4 Parity, NorthStar, ONNX)
# ---------------------------------------------------------------------------


def _materialize_int4_dequant_ckpt(
    repo_root: Path, src_ckpt: Path, dst_ckpt: Path
) -> int:
    mvd_dir = str(repo_root / "tmp-train" / "mvd")
    if mvd_dir not in sys.path:
        sys.path.insert(0, mvd_dir)
    import torch
    from uni_omr_v2 import QATInt4ControllerV2, UniOMRv2Config, UniOMRv2Net, fake_quant_int4_ste

    ck = torch.load(str(src_ckpt), map_location="cpu", weights_only=True)
    cfg = UniOMRv2Config(**ck["cfg"])
    model = UniOMRv2Net(cfg)
    model.load_state_dict(ck["model"], strict=False)
    ctrl = QATInt4ControllerV2(model, block_size=32, enabled=True)
    with torch.no_grad():
        for mod in ctrl._targets:
            mod.weight.copy_(fake_quant_int4_ste(mod.weight, block_size=32))
    ctrl.detach()
    ck["model"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    tmp = dst_ckpt.with_suffix(".pt.tmp")
    torch.save(ck, str(tmp))
    os.replace(tmp, dst_ckpt)
    _log(f"[INT4-dequant] Quantized {ctrl.num_layers} Linear layers -> {dst_ckpt}")
    return int(ctrl.num_layers)


def run_phase3_evals(
    *,
    python_exe: str,
    repo_root: Path,
    eval_ckpt: Path,
    eval_out_dir: Path,
    northstar_out_dir: Path,
    onnx_out_dir: Path,
    suite31_manifest: Path,
    northstar_manifest: Path,
    hf_token: str,
    profile: GpuTuningProfile,
) -> Dict[str, Any]:
    eval_out_dir.mkdir(parents=True, exist_ok=True)
    northstar_out_dir.mkdir(parents=True, exist_ok=True)
    onnx_out_dir.mkdir(parents=True, exist_ok=True)

    aser_bin, _eharness_bin = _ensure_rust_eval_binaries(repo_root)
    eval_script = str(repo_root / "tmp-train" / "mvd" / "eval_uni_suite31.py")
    onnx_script = str(repo_root / "tmp-train" / "mvd" / "export_uni_v2_onnx.py")
    northstar_script = str(
        repo_root
        / "experiments"
        / "2026-11-26T14-40-eval-northstar-iter12-v2"
        / "run_eval.py"
    )

    eval_fast_out = eval_out_dir / "iter15_edge85m_stage2_20k_ema_latest_1408_fsa_mtp.json"
    eval_ilms_out = eval_out_dir / "iter15_edge85m_stage2_20k_ema_latest_1408_fsa_mtp_ilms.json"
    int4_ckpt = eval_ckpt.parent / "uni_omr_v2_ema_latest_int4dequant.pt"
    eval_int4_out = (
        eval_out_dir / "iter15_edge85m_stage2_20k_ema_latest_int4dequant_1408_fsa_mtp.json"
    )

    # 3.1 suite31_hw Fast
    _log("=== [Phase 3.1] suite31_hw Fast (1408px/3584, FSA+MTP-4, mem=4096) ===")
    _run_cmd(
        [
            python_exe,
            "-u",
            eval_script,
            "--ckpt",
            str(eval_ckpt),
            "--manifest",
            str(suite31_manifest),
            "--max-dim",
            "1408",
            "--max-len",
            "3584",
            "--max-mem-tokens",
            "4096",
            "--use-grammar",
            "--use-mtp-spec",
            "--bf16",
            "--out",
            str(eval_fast_out),
        ],
        cwd=repo_root,
    )

    # 3.2 suite31_hw +ILMS
    _log("=== [Phase 3.2] suite31_hw +ILMS (0.40/0.85, mem=4096) ===")
    _run_cmd(
        [
            python_exe,
            "-u",
            eval_script,
            "--ckpt",
            str(eval_ckpt),
            "--manifest",
            str(suite31_manifest),
            "--max-dim",
            "1408",
            "--max-len",
            "3584",
            "--max-mem-tokens",
            "4096",
            "--use-grammar",
            "--use-mtp-spec",
            "--ilms-alpha",
            "0.40",
            "--ilms-conf-thr",
            "0.85",
            "--bf16",
            "--out",
            str(eval_ilms_out),
        ],
        cwd=repo_root,
    )

    # 3.3 INT4-dequant parity check
    _log("=== [Phase 3.3] Materializing INT4-dequant checkpoint & running parity eval ===")
    num_int4_layers = _materialize_int4_dequant_ckpt(repo_root, eval_ckpt, int4_ckpt)
    _run_cmd(
        [
            python_exe,
            "-u",
            eval_script,
            "--ckpt",
            str(int4_ckpt),
            "--manifest",
            str(suite31_manifest),
            "--max-dim",
            "1408",
            "--max-len",
            "3584",
            "--max-mem-tokens",
            "4096",
            "--use-grammar",
            "--use-mtp-spec",
            "--bf16",
            "--out",
            str(eval_int4_out),
        ],
        cwd=repo_root,
    )

    # 3.4 125-page OMR-NorthStar
    _log("=== [Phase 3.4] 125-page OMR-NorthStar (--neg-control-gate) ===")
    _run_cmd(
        [
            python_exe,
            "-u",
            northstar_script,
            "--ckpt",
            str(eval_ckpt),
            "--manifest",
            str(northstar_manifest),
            "--aser-bin",
            str(aser_bin),
            "--system-name",
            "uni-omr-iter15-edge85m-20k-ema-last",
            "--out-dir",
            str(northstar_out_dir),
            "--device",
            "cuda",
            "--bf16",
            "--neg-control-gate",
            "--resume",
        ],
        cwd=repo_root,
    )

    # 3.5 WebGPU INT4 ONNX export & < 55 MB gate
    _log(f"=== [Phase 3.5] WebGPU INT4 ONNX Export (< {WEBGPU_MAX_MB:.1f} MB gate) ===")
    _run_cmd(
        [
            python_exe,
            "-u",
            onnx_script,
            "--ckpt",
            str(eval_ckpt),
            "--preset",
            "edge85m",
            "--targets",
            "webgpu",
            "--fp16",
            "--nbits-block-size",
            "32",
            "--out-dir",
            str(onnx_out_dir),
        ],
        cwd=repo_root,
    )

    webgpu_dir = onnx_out_dir / "webgpu"
    total_bytes = sum(p.stat().st_size for p in webgpu_dir.rglob("*") if p.is_file())
    total_mb = round(total_bytes / float(1024 * 1024), 2)
    webgpu_gate_pass = total_mb < WEBGPU_MAX_MB
    _log(
        f"WebGPU INT4 total artifact size: {total_mb:.2f} MB "
        f"(gate: < {WEBGPU_MAX_MB:.1f} MB -> {'PASS' if webgpu_gate_pass else 'FAIL'})"
    )

    ns_report_path = northstar_out_dir / "report.json"
    summary: Dict[str, Any] = {
        "exp_id": EXP_ID,
        "run_tag": RUN_TAG,
        "completed_utc": _utc_now(),
        "gpu_profile": dataclasses.asdict(profile),
        "int4_quantized_layers": num_int4_layers,
        "webgpu_int4_mb": total_mb,
        "webgpu_gate_pass": webgpu_gate_pass,
        "artifacts": {
            "suite31_fast": str(eval_fast_out),
            "suite31_ilms": str(eval_ilms_out),
            "suite31_int4": str(eval_int4_out),
            "northstar_report": str(ns_report_path),
        },
    }
    summary_path = eval_out_dir / f"{RUN_TAG}_summary.json"
    _atomic_write_json(summary_path, summary)

    # Upload all final evals + ledger to HF
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    uploads = [
        (eval_fast_out, f"{HF_EVAL_PREFIX}_suite31_hw.json"),
        (eval_ilms_out, f"{HF_EVAL_PREFIX}_suite31_hw_ilms.json"),
        (eval_int4_out, f"{HF_EVAL_PREFIX}_suite31_hw_int4dequant.json"),
        (summary_path, f"{HF_EVAL_PREFIX}_summary.json"),
        (summary_path, HF_RUN_LEDGER_PATH),
    ]
    if ns_report_path.is_file():
        uploads.append((ns_report_path, f"{HF_EVAL_PREFIX}_northstar.json"))

    for local_p, repo_p in uploads:
        if local_p.is_file():
            _log(f"Uploading {local_p.name} -> HF {repo_p}...")
            api.upload_file(
                path_or_fileobj=str(local_p),
                path_in_repo=repo_p,
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                commit_message=f"eval({RUN_TAG}): upload {local_p.name}",
            )

    if not webgpu_gate_pass:
        raise RuntimeError(
            f"WebGPU INT4 ONNX export exceeded {WEBGPU_MAX_MB:.1f} MB gate: {total_mb:.2f} MB"
        )
    return summary


# ---------------------------------------------------------------------------
# 8. CPU Self-Test (`--dry-run`)
# ---------------------------------------------------------------------------


def run_dry_run_self_test(repo_root: Path) -> None:
    """Verify GPU auto-tuning, Windows->Colab path remapping, commands, and 5-min syncer on CPU."""
    _log("=== Running CPU --dry-run self-test ===")

    # 1. Verify GPU auto-tuning tiers
    p_h100 = select_gpu_profile(vram_gb=80.0, device_name="NVIDIA H100 80GB HBM3")
    p_a100 = select_gpu_profile(vram_gb=40.0, device_name="NVIDIA A100-SXM4-40GB")
    p_l4 = select_gpu_profile(vram_gb=22.5, device_name="NVIDIA L4")
    assert (
        p_h100.tier,
        p_h100.batch_size,
        p_h100.grad_accum,
        p_h100.strip_batch_size,
        p_h100.strip_grad_accum,
        p_h100.use_grad_checkpoint,
        p_h100.num_workers,
    ) == (
        "H100-80GB",
        16,
        1,
        64,
        1,
        False,
        8,
    ), f"Unexpected H100 profile: {p_h100}"
    assert (p_a100.tier, p_a100.batch_size, p_a100.grad_accum, p_a100.use_grad_checkpoint) == (
        "A100-40GB",
        4,
        2,
        True,
    ), f"Unexpected A100 profile: {p_a100}"
    assert (p_l4.tier, p_l4.batch_size, p_l4.grad_accum, p_l4.use_grad_checkpoint) == (
        "L4-24GB",
        2,
        4,
        True,
    ), f"Unexpected L4 profile: {p_l4}"
    _log(
        f"[OK] GPU tiers verified: H100(bs={p_h100.batch_size},ga={p_h100.grad_accum},ckpt={p_h100.use_grad_checkpoint}), "
        f"A100(bs={p_a100.batch_size},ga={p_a100.grad_accum},ckpt={p_a100.use_grad_checkpoint}), "
        f"L4(bs={p_l4.batch_size},ga={p_l4.grad_accum},ckpt={p_l4.use_grad_checkpoint})"
    )

    with tempfile.TemporaryDirectory(prefix="iter15_colab_dryrun_") as td:
        tmp_root = Path(td)
        fake_data = tmp_root / "data"
        fake_run = tmp_root / "run"
        fake_data.mkdir(parents=True, exist_ok=True)
        fake_run.mkdir(parents=True, exist_ok=True)

        # 2. Verify Windows -> Linux path remapper + JSONL manifest remapper
        sample_img = r"D:\tmp\unlabeled-10k-real\rism_hw\page_001.png"
        sample_bek = r"D:\tmp\iter15_v2a\bekerns\page_001_s0.bekern"
        remapped_img = remap_windows_path(sample_img, data_root=Path("/content/data"))
        remapped_bek = remap_windows_path(sample_bek, data_root=Path("/content/data"))
        assert remapped_img == "/content/data/unlabeled-10k-real/rism_hw/page_001.png", remapped_img
        assert remapped_bek == "/content/data/iter15_v2a/bekerns/page_001_s0.bekern", remapped_bek

        src_jsonl = fake_data / "sample_manifest.jsonl"
        dst_jsonl = fake_data / "sample_manifest_colab.jsonl"
        src_jsonl.write_text(
            json.dumps(
                {
                    "image": sample_img,
                    "bekerns": [sample_bek],
                    "domain": "real_hw",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        stats = remap_jsonl_manifest(src_jsonl, dst_jsonl, data_root=Path("/content/data"))
        assert stats["total_rows"] == 1 and stats["kept_rows"] == 1 and stats["remapped_fields"] == 2, stats
        parsed = json.loads(dst_jsonl.read_text(encoding="utf-8").strip())
        assert parsed["image"] == "/content/data/unlabeled-10k-real/rism_hw/page_001.png"
        assert parsed["bekerns"] == ["/content/data/iter15_v2a/bekerns/page_001_s0.bekern"]
        _log(f"[OK] Path remapper verified: {stats}")

        # 3. Verify Phase 0 / 1 / 2 H100 command flags
        c0 = build_phase0_cmd(
            python_exe=sys.executable,
            repo_root=repo_root,
            profile=p_h100,
            rism_dirs=[fake_data / "rism_a"],
            printed_manifest=dst_jsonl,
            out_dir=fake_run / "phase0",
        )
        assert "--no-grad-checkpoint" in c0 and "--save-every-minutes" in c0 and "5.0" in c0
        assert "--n-axial-layers" in c0 and c0[c0.index("--n-axial-layers") + 1] == "0"
        assert "--n-global-layers" in c0 and c0[c0.index("--n-global-layers") + 1] == "6"
        assert "--steps" in c0 and c0[c0.index("--steps") + 1] == "1500"
        assert "--batch-size" in c0 and c0[c0.index("--batch-size") + 1] == "16"
        assert "--grad-accum" in c0 and c0[c0.index("--grad-accum") + 1] == "1"

        c1 = build_phase1_cmd(
            python_exe=sys.executable,
            repo_root=repo_root,
            profile=p_h100,
            phase0_ckpt=fake_run / "phase0" / "uni_omr_v2_vjepa_last.pt",
            printed_strip=fake_data / "strips",
            hw_strips=[fake_data / "hw_strips"],
            out_dir=fake_run / "phase1",
        )
        assert "--grad-checkpoint" not in c1
        assert "--n-axial-layers" in c1 and c1[c1.index("--n-axial-layers") + 1] == "0"
        assert "--n-global-layers" in c1 and c1[c1.index("--n-global-layers") + 1] == "6"
        assert "--steps" in c1 and c1[c1.index("--steps") + 1] == "2000"
        assert "--strip-batch-size" in c1 and c1[c1.index("--strip-batch-size") + 1] == "64"
        assert "--strip-grad-accum" in c1 and c1[c1.index("--strip-grad-accum") + 1] == "1"
        assert "--num-workers" in c1 and c1[c1.index("--num-workers") + 1] == "8"
        assert "--eval-ema-only" in c1
        assert "--save-every-minutes" in c1

        c2 = build_phase2_cmd(
            python_exe=sys.executable,
            repo_root=repo_root,
            profile=p_h100,
            phase1_ckpt=fake_run / "phase1" / "uni_omr_v2_ema_best.pt",
            stage2_manifest=dst_jsonl,
            rism_style_dir=fake_data / "rism_a",
            teacher_ckpt=fake_data / "teacher.pt",
            out_dir=fake_run / "phase2",
        )
        assert "--grad-checkpoint" not in c2
        assert "--n-axial-layers" in c2 and c2[c2.index("--n-axial-layers") + 1] == "0"
        assert "--n-global-layers" in c2 and c2[c2.index("--n-global-layers") + 1] == "6"
        assert "--steps" in c2 and c2[c2.index("--steps") + 1] == "12000"
        assert "--batch-size" in c2 and c2[c2.index("--batch-size") + 1] == "16"
        assert "--grad-accum" in c2 and c2[c2.index("--grad-accum") + 1] == "1"
        assert "--prog-res-schedule" in c2 and c2[c2.index("--prog-res-schedule") + 1] == STAGE2_PROG_RES_SCHEDULE
        assert "--cass-start-frac" in c2 and c2[c2.index("--cass-start-frac") + 1] == "0.35"
        assert "--eval-ema-only" in c2
        assert "--num-workers" in c2 and c2[c2.index("--num-workers") + 1] == "8"
        assert "--fixed-hw-buckets" in c2
        assert "--stratified-preset" in c2 and c2[c2.index("--stratified-preset") + 1] == "v3"
        assert "--expr-token-loss-weight" in c2 and c2[c2.index("--expr-token-loss-weight") + 1] == "2.5"
        assert "--tok-dropout" in c2 and c2[c2.index("--tok-dropout") + 1] == "0.12"
        assert "--adaptive-ink-target" in c2
        assert "--qat-int4" in c2 and "--qat-int4-start-frac" in c2 and "--qat-int4-block-size" in c2
        assert "--save-ema-milestones" in c2 and c2[c2.index("--save-ema-milestones") + 1] == "4000"
        assert "--save-every-minutes" in c2
        _log("[OK] Phase 0/1/2 H100 CLI commands verified.")

        # 4. Verify 5-minute background HF syncer (mtime dedup + atomic copy + retry + final sync)
        p2_dir = fake_run / "phase2"
        p2_dir.mkdir(parents=True, exist_ok=True)
        uploaded_records: List[Tuple[str, int]] = []
        fail_once_done = {"fired": False}

        def _fake_upload(tmp_path: Path, repo_path: str, _msg: str) -> None:
            assert tmp_path.is_file() and tmp_path.name.endswith(".tmp")
            if not fail_once_done["fired"] and repo_path.endswith("uni_omr_v2_latest.pt"):
                fail_once_done["fired"] = True
                raise ConnectionError("simulated transient HF upload error")
            uploaded_records.append((repo_path, tmp_path.stat().st_size))

        syncer = HFCheckpointSyncer(
            run_root=fake_run,
            interval_seconds=0.1,
            max_retries=3,
            base_backoff_s=0.02,
            dry_run=False,
            upload_fn=_fake_upload,
        )
        syncer.register_dir("phase2", p2_dir)
        syncer.start()
        try:
            _atomic_write_json(fake_run / "phase_state.json", {"active_phase": 2})
            (p2_dir / "uni_omr_v2_latest.pt").write_bytes(b"ckpt-v1")
            (p2_dir / "uni_omr_v2_ema_step4000.pt").write_bytes(b"ema-4k")
            (p2_dir / "ignored_scratch.bin").write_bytes(b"ignore-me")
            (p2_dir / "uni_omr_v2_latest.pt.tmp").write_bytes(b"partial-tmp")

            time.sleep(0.35)
            # Second sync with unchanged files should upload 0 new files
            count_before = len(uploaded_records)
            assert syncer.sync_once(reason="noop-check") == 0
            assert len(uploaded_records) == count_before

            # Mutating latest.pt (new size + mtime) should trigger re-upload on final flush
            (p2_dir / "uni_omr_v2_latest.pt").write_bytes(b"ckpt-v2-updated")
        finally:
            final_n = syncer.stop_and_final_sync()

        assert final_n == 1, f"Expected 1 file on final flush, got {final_n}"
        repo_paths = [rp for rp, _ in uploaded_records]
        assert f"{HF_CKPT_PREFIX}/phase_state.json" in repo_paths
        assert f"{HF_CKPT_PREFIX}/phase2/uni_omr_v2_latest.pt" in repo_paths
        assert f"{HF_CKPT_PREFIX}/phase2/uni_omr_v2_ema_step4000.pt" in repo_paths
        assert all("ignored_scratch" not in rp and not rp.endswith(".tmp") for rp in repo_paths)
        assert fail_once_done["fired"] and syncer.error_count == 1
        _log(
            f"[OK] HFCheckpointSyncer verified ({len(uploaded_records)} uploads, "
            f"backoff retry passed, mtime dedup passed, final flush passed)."
        )

    _log("=== CPU --dry-run self-test PASSED ===")


# ---------------------------------------------------------------------------
# 9. Main Orchestrator
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Colab H100/A100/L4 Recoverable 4-Phase Runner for Iter 15"
    )
    p.add_argument("--dry-run", action="store_true", help="Run CPU self-test and exit")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    p.add_argument("--stage0-steps", type=int, default=STAGE0_STEPS_DEFAULT)
    p.add_argument("--stage1-steps", type=int, default=STAGE1_STEPS_DEFAULT)
    p.add_argument("--stage2-steps", type=int, default=STAGE2_STEPS_DEFAULT)
    p.add_argument("--stage2-warmup", type=int, default=STAGE2_WARMUP_DEFAULT)
    p.add_argument("--sync-interval-seconds", type=float, default=300.0)
    p.add_argument("--skip-phase0", action="store_true")
    p.add_argument("--skip-phase1", action="store_true")
    p.add_argument("--skip-phase2", action="store_true")
    p.add_argument("--skip-phase3", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    repo_root = _resolve_repo_root()
    if args.dry_run:
        run_dry_run_self_test(repo_root)
        print(f"[bootstrap:{RUN_TAG}] Finished rc=0", flush=True)
        return 0

    _ensure_python_deps(include_onnx=True)
    hf_token = _read_hf_token(required=True)

    profile = select_gpu_profile()
    _log(
        f"Auto-tuned GPU profile: tier={profile.tier}, device={profile.device_name}, "
        f"vram={profile.vram_gb:.2f} GB, bs={profile.batch_size}, ga={profile.grad_accum}, "
        f"grad_ckpt={profile.use_grad_checkpoint}, bf16={profile.use_bf16}, "
        f"no_cudnn={profile.use_no_cudnn}"
    )

    run_root = args.run_root
    phase_dirs = {
        "phase0": run_root / "phase0-stage0-vjepa",
        "phase1": run_root / "phase1-stage1-strip4k",
        "phase2": run_root / "phase2-stage2-20k",
    }
    eval_out_dir = run_root / "phase3-evals"
    northstar_out_dir = run_root / "phase3-northstar"
    onnx_out_dir = phase_dirs["phase2"] / "onnx_webgpu_int4"
    for d in (*phase_dirs.values(), eval_out_dir, northstar_out_dir, onnx_out_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Restore prior HF state/checkpoints before starting syncer
    state = restore_from_hf_if_present(
        hf_token=hf_token, run_root=run_root, phase_dirs=phase_dirs
    )
    completed = set(int(x) for x in state.get("completed_phases", []))
    state_file = run_root / "phase_state.json"

    def _save_state(active_phase: int) -> None:
        state["active_phase"] = active_phase
        state["completed_phases"] = sorted(completed)
        state["gpu_tier"] = profile.tier
        state["updated_utc"] = _utc_now()
        _atomic_write_json(state_file, state)

    assets = pull_datasets_and_teacher(
        hf_token=hf_token, data_root=args.data_root, repo_root=repo_root
    )

    syncer = HFCheckpointSyncer(
        run_root=run_root,
        hf_repo_id=HF_REPO_ID,
        hf_ckpt_prefix=HF_CKPT_PREFIX,
        hf_token=hf_token,
        interval_seconds=args.sync_interval_seconds,
    )
    for tag, pdir in phase_dirs.items():
        syncer.register_dir(tag, pdir)
    syncer.start()

    try:
        # -------------------------------------------------------------------
        # Phase 0: Stage-0 V-JEPA 2.1 + VICReg Pretraining (3,000 steps)
        # -------------------------------------------------------------------
        p0_dir = phase_dirs["phase0"]
        if not args.skip_phase0 and 0 not in completed:
            _save_state(0)
            syncer.sync_once(reason="phase0-start")
            p0_resume = p0_dir / "uni_omr_v2_vjepa_autosave.pt"
            if not p0_resume.is_file():
                p0_resume = None
            cmd0 = build_phase0_cmd(
                python_exe=sys.executable,
                repo_root=repo_root,
                profile=profile,
                rism_dirs=[assets["rism_dir_a"], assets["rism_dir_b"]],
                printed_manifest=assets["printed_manifest"],
                out_dir=p0_dir,
                steps=args.stage0_steps,
                resume_ckpt=p0_resume,
            )
            _log("=== [Iter 15 Phase 0] Stage-0 V-JEPA+VICReg ===")
            _run_cmd(cmd0, cwd=repo_root)
            completed.add(0)
            _save_state(1)
            syncer.sync_once(reason="phase0-done")
        else:
            _log("=== [Iter 15 Phase 0] Skipped (completed or --skip-phase0) ===")

        p0_last = p0_dir / "uni_omr_v2_vjepa_last.pt"
        p0_best = p0_dir / "uni_omr_v2_vjepa_best.pt"
        p0_auto = p0_dir / "uni_omr_v2_vjepa_autosave.pt"
        phase0_ckpt = p0_last if p0_last.is_file() else (p0_best if p0_best.is_file() else p0_auto)

        # -------------------------------------------------------------------
        # Phase 1: Stage-1 Strip Warmup (4,000 steps)
        # -------------------------------------------------------------------
        p1_dir = phase_dirs["phase1"]
        if not args.skip_phase1 and 1 not in completed:
            _save_state(1)
            syncer.sync_once(reason="phase1-start")
            p1_resume = p1_dir / "uni_omr_v2_latest.pt"
            if not p1_resume.is_file():
                p1_resume = None
            cmd1 = build_phase1_cmd(
                python_exe=sys.executable,
                repo_root=repo_root,
                profile=profile,
                phase0_ckpt=phase0_ckpt,
                printed_strip=assets["printed_strip"],
                hw_strips=[assets["hw_strip"], assets["hw_strip_v06"]],
                out_dir=p1_dir,
                steps=args.stage1_steps,
                resume_ckpt=p1_resume,
            )
            _log("=== [Iter 15 Phase 1] Stage-1 Strip Warmup ===")
            _run_cmd(cmd1, cwd=repo_root)
            completed.add(1)
            _save_state(2)
            syncer.sync_once(reason="phase1-done")
        else:
            _log("=== [Iter 15 Phase 1] Skipped (completed or --skip-phase1) ===")

        p1_best = p1_dir / "uni_omr_v2_ema_best.pt"
        p1_latest = p1_dir / "uni_omr_v2_ema_latest.pt"
        phase1_ckpt = p1_best if p1_best.is_file() else p1_latest

        # -------------------------------------------------------------------
        # Phase 2: Stage-2 Full-Page (20,000 steps)
        # -------------------------------------------------------------------
        p2_dir = phase_dirs["phase2"]
        if not args.skip_phase2 and 2 not in completed:
            if assets["stage2_manifest"] is None or not Path(assets["stage2_manifest"]).is_file():
                raise FileNotFoundError(
                    "Stage-2 manifest uni_fullpage_iter15_v2a.jsonl not found after HF dataset pull."
                )
            _save_state(2)
            syncer.sync_once(reason="phase2-start")
            p2_resume = p2_dir / "uni_omr_v2_latest.pt"
            if not p2_resume.is_file():
                p2_resume = None
            cmd2 = build_phase2_cmd(
                python_exe=sys.executable,
                repo_root=repo_root,
                profile=profile,
                phase1_ckpt=phase1_ckpt,
                stage2_manifest=assets["stage2_manifest"],
                rism_style_dir=assets["rism_dir_a"],
                teacher_ckpt=assets["teacher_ckpt"],
                out_dir=p2_dir,
                steps=args.stage2_steps,
                warmup_steps=args.stage2_warmup,
                resume_ckpt=p2_resume,
            )
            _log("=== [Iter 15 Phase 2] Stage-2 Full-Page (20,000 steps) ===")
            _run_cmd(cmd2, cwd=repo_root)
            completed.add(2)
            _save_state(3)
            syncer.sync_once(reason="phase2-done")
        else:
            _log("=== [Iter 15 Phase 2] Skipped (completed or --skip-phase2) ===")

        # -------------------------------------------------------------------
        # Phase 3: Automatic suite31_hw, OMR-NorthStar, WebGPU INT4 ONNX
        # -------------------------------------------------------------------
        p2_ema_latest = p2_dir / "uni_omr_v2_ema_latest.pt"
        p2_ema_best = p2_dir / "uni_omr_v2_ema_best.pt"
        eval_ckpt = p2_ema_latest if p2_ema_latest.is_file() else p2_ema_best

        if not args.skip_phase3 and 3 not in completed:
            if not eval_ckpt.is_file():
                raise FileNotFoundError(f"Phase 2 EMA checkpoint not found at {eval_ckpt}")
            _save_state(3)
            run_phase3_evals(
                python_exe=sys.executable,
                repo_root=repo_root,
                eval_ckpt=eval_ckpt,
                eval_out_dir=eval_out_dir,
                northstar_out_dir=northstar_out_dir,
                onnx_out_dir=onnx_out_dir,
                suite31_manifest=assets["suite31_manifest"],
                northstar_manifest=assets["northstar_manifest"],
                hf_token=hf_token,
                profile=profile,
            )
            completed.add(3)
            _save_state(4)
        else:
            _log("=== [Iter 15 Phase 3] Skipped (completed or --skip-phase3) ===")

        _log("=== [Iter 15 Colab H100] All requested phases completed cleanly! ===")
        print(f"[bootstrap:{RUN_TAG}] Finished rc=0", flush=True)
        return 0
    except Exception:
        traceback.print_exc()
        print(f"[bootstrap:{RUN_TAG}] Finished rc=1", flush=True)
        return 1
    finally:
        syncer.stop_and_final_sync()
        _purge_tmp_token_file()


if __name__ == "__main__":
    raise SystemExit(main())
