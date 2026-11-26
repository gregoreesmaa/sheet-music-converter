#!/usr/bin/env python3
"""Parallel Kaggle 2x T4 DAG worker for `experiments/overnight_queue/queue.json`.

Cooperates cleanly with `experiments/overnight_queue/run_queue_colab.py` on the
same `queue.json` without colliding:
  - Only picks verified T4-safe (FP16/FP32, non-BF16-only) experiments whose
    `depends_on` predecessors are all satisfied (`COMPLETED` in `queue.json`,
    eval JSON exists on HF `evals/<dep>.json`, or `OUTCOME.md` exists locally).
  - Defaults to reverse priority order (`--order reverse`) so Kaggle works from
    the tail of the queue (`ov-39`, `ov-37`, `ov-38`, `ov-21`, `ov-34`, ...)
    while Colab works from the front (`ov-36`, `ov-35`, `ov-34`, `ov-13`, ...).
  - Always re-reads `queue.json` fresh from disk under an atomic cross-process
    lockfile (`.queue.lock`) before claiming `status="RUNNING"` with
    `"provider": "kaggle"` and before writing `COMPLETED`/`FAILED`.
  - Checks HF `evals/<exp_id>.json` before pushing any kernel, and verifies
    that `evals/<exp_id>.json` exists on HF after kernel completion.
  - Builds one shared lightweight code bundle (`tmp-train/mvd/` + Rust eval
    crates + `mscz-rs`) and uploads it once to HF `code/ov_mvd_bundle.tar.gz`
    so Kaggle kernels run the exact local working tree.
  - Mounts `"dataset_sources": ["gregoreesmaa/omr-hf-auth"]` in strictly
    private (`"is_private": true`), GPU-enabled (`"enable_gpu": true`),
    internet-enabled (`"enable_internet": true`) Kaggle kernels, reading
    `HF_TOKEN` from `/kaggle/input/**/hf_token.txt` without ever printing it.
  - Enforces T4-safe precision (never `--bf16` on `cc < 8.0`; strips/replaces
    `--bf16` and relies on `cc[0] >= 8` guards for FP32/FP16 on Turing T4).
  - Decodes Kaggle's JSON-array `<slug>.log` into plain text under
    `D:\\tmp\\overnight-queue-logs\\<exp_id>.kaggle.log`.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait as fut_wait
from datetime import datetime, timezone
import hashlib
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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
HF_BUNDLE_PATH = "code/ov_mvd_bundle.tar.gz"
KAGGLE_USERNAME = "gregoreesmaa"
KAGGLE_HF_AUTH_DATASET = "gregoreesmaa/omr-hf-auth"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUEUE_PATH = REPO_ROOT / "experiments" / "overnight_queue" / "queue.json"
DEFAULT_LOG_DIR = Path(r"D:\tmp\overnight-queue-logs")
DEFAULT_KERNEL_WORK_DIR = Path(r"D:\tmp\kaggle-queue-kernels")
DEFAULT_KAGGLE_EXE = REPO_ROOT / ".venv" / "Scripts" / "kaggle.exe"

# Verified T4-compatible (cc=7.5, FP32/FP16 safe, never BF16-only, fits 14.56 GiB T4)
# experiment short tags. All of these either use pure FP32 image processing (ov-21)
# or explicitly guard `use_bf16 = bool(cc[0] >= 8)` and run pure inference/eval or
# light per-page/small-step adaptation cleanly in FP32/FP16 on T4.
VERIFIED_T4_SAFE_TAGS: Dict[str, str] = {
    "ov-21": "pure FP32 flood-fill paper_gate border sweep (290 pages, no training, no Rust)",
    "ov-34": "inference-only measure-beam x header-guard sweep (cc>=8 guard -> FP32 on T4)",
    "ov-34b": "inference-only budgeted logprob measure-beam sweep (cc>=8 guard -> FP32 on T4)",
    "ov-35": "inference-only max_len x resolution sweep (cc>=8 guard -> FP32 on T4)",
    "ov-35b": "inference-only lower-res x tight-maxlen sweep (cc>=8 guard -> FP32 on T4)",
    "ov-36": "inference-only HW multi-view TTA + voting (cc>=8 guard -> FP32 on T4)",
    "ov-36b": "inference-only multiscale logprob FSA TTA (cc>=8 guard -> FP32 on T4)",
    "ov-37": "6-arm RLVR GRPO sweep (cc>=8 guard -> FP32 on T4, pure Python)",
    "ov-37b": "RLVR GRPO scale200 len512 sweep (cc>=8 guard -> FP16+GradScaler on T4, pure Python)",
    "ov-38": "SCRIBE/DocTTT per-manuscript TTA sweep (cc>=8 guard -> FP32 on T4, pure Python)",
    "ov-39": "RISM pseudo-label + 60-step FT sweep (cc>=8 guard -> FP32 on T4, pure Python)",
}

PRINT_LOCK = threading.Lock()
THREAD_QUEUE_LOCK = threading.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log_console(msg: str) -> None:
    with PRINT_LOCK:
        print(f"[{_utc_now()}] {msg}", flush=True)


def _short_tag(exp_id: str) -> str:
    m = re.search(r"(ov-\d+[a-z]*)", exp_id)
    if m:
        return m.group(1)
    slug = re.sub(r"[^a-z0-9-]+", "-", exp_id.lower()).strip("-")
    return slug[:24] or "ov-run"


def _kaggle_slug_for(exp_id: str) -> str:
    """Deterministic short Kaggle kernel slug (lowercase a-z0-9-, e.g. omr-ov-37)."""
    tag = _short_tag(exp_id)
    slug = f"omr-{tag}"
    slug = re.sub(r"[^a-z0-9-]+", "-", slug.lower())
    slug = re.sub(r"-+", "-", slug).strip("-")
    if len(slug) > 48:
        slug = slug[:48].rstrip("-")
    return slug


def _extract_pushed_kernel_ref(push_output: str) -> Optional[str]:
    """Parse canonical `<user>/<slug>` from `kaggle kernels push` output URL."""
    m = re.search(
        r"https?://(?:www\.)?kaggle\.com/code/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)",
        push_output,
    )
    if m:
        return m.group(1).rstrip(".,;)")
    return None


def _acquire_file_lock(lock_path: Path, timeout_s: float = 30.0) -> int:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.write(fd, f"pid={os.getpid()} ts={_utc_now()}\n".encode("utf-8"))
            return fd
        except FileExistsError:
            # Break stale lock older than 120s
            try:
                age = time.time() - lock_path.stat().st_mtime
                if age > 120.0:
                    lock_path.unlink(missing_ok=True)
                    continue
            except OSError:
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for lock {lock_path}")
            time.sleep(0.1)


def _release_file_lock(fd: int, lock_path: Path) -> None:
    try:
        os.close(fd)
    except OSError:
        pass
    try:
        lock_path.unlink(missing_ok=True)
    except OSError:
        pass


def _load_queue_unlocked(path: Path) -> List[Dict[str, Any]]:
    for attempt in range(10):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                raise ValueError(f"Expected JSON list in {path}")
            return data
        except (PermissionError, json.JSONDecodeError):
            if attempt == 9:
                raise
            time.sleep(0.15)
    raise RuntimeError(f"Failed to read {path}")


def _save_queue_atomic_unlocked(path: Path, queue: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.kaggle.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(queue, indent=2) + "\n", encoding="utf-8")
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 9:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(0.15)


def _read_queue(queue_path: Path) -> List[Dict[str, Any]]:
    lock_path = queue_path.with_name(".queue.lock")
    with THREAD_QUEUE_LOCK:
        fd = _acquire_file_lock(lock_path)
        try:
            return _load_queue_unlocked(queue_path)
        finally:
            _release_file_lock(fd, lock_path)


def _claim_experiment_atomic(
    queue_path: Path,
    exp_id: str,
    claim_updates: Dict[str, Any],
) -> bool:
    """Atomically re-read queue.json and claim exp_id ONLY if status is QUEUED."""
    lock_path = queue_path.with_name(".queue.lock")
    with THREAD_QUEUE_LOCK:
        fd = _acquire_file_lock(lock_path)
        try:
            queue = _load_queue_unlocked(queue_path)
            for row in queue:
                if str(row.get("id")) == exp_id:
                    st = str(row.get("status", "QUEUED")).upper()
                    if st in {"RUNNING", "COMPLETED"}:
                        return False
                    row.update(claim_updates)
                    _save_queue_atomic_unlocked(queue_path, queue)
                    return True
            return False
        finally:
            _release_file_lock(fd, lock_path)


def _update_experiment_record(
    queue_path: Path,
    exp_id: str,
    updates: Dict[str, Any],
) -> None:
    """Re-read queue.json fresh under lock and update only exp_id."""
    lock_path = queue_path.with_name(".queue.lock")
    with THREAD_QUEUE_LOCK:
        fd = _acquire_file_lock(lock_path)
        try:
            queue = _load_queue_unlocked(queue_path)
            for row in queue:
                if str(row.get("id")) == exp_id:
                    row.update(updates)
                    break
            _save_queue_atomic_unlocked(queue_path, queue)
        finally:
            _release_file_lock(fd, lock_path)


def _resolve_hf_token() -> Optional[str]:
    env_tok = os.environ.get("HF_TOKEN", "").strip()
    if env_tok:
        return env_tok
    default_tok = Path.home() / ".cache" / "huggingface" / "token"
    if default_tok.is_file():
        tok = default_tok.read_text(encoding="utf-8").strip()
        if tok:
            return tok
    return None


def _hf_file_exists(path_in_repo: str, token: Optional[str] = None) -> bool:
    try:
        from huggingface_hub import HfApi  # type: ignore

        api = HfApi(token=token) if token else HfApi()
        return bool(
            api.file_exists(
                HF_REPO_ID,
                path_in_repo,
                repo_type="dataset",
            )
        )
    except Exception as exc:
        _log_console(f"[WARN] HF existence check failed for {path_in_repo}: {exc}")
        return False


def _local_outcome_exists(exp_id: str) -> bool:
    return (REPO_ROOT / "experiments" / exp_id / "OUTCOME.md").is_file()


def _is_t4_eligible(
    exp: Dict[str, Any],
    *,
    allow_tags: Optional[Set[str]] = None,
) -> Tuple[bool, str]:
    exp_id = str(exp.get("id", ""))
    tag = _short_tag(exp_id)
    script_path = REPO_ROOT / str(exp.get("script", ""))
    if not script_path.is_file():
        return False, f"missing script {exp.get('script')}"

    if allow_tags is not None:
        if tag not in allow_tags and exp_id not in allow_tags:
            return False, "not in --only-ids filter"

    if tag in VERIFIED_T4_SAFE_TAGS:
        return True, VERIFIED_T4_SAFE_TAGS[tag]

    # If explicitly requested via --only-ids, inspect script for hard BF16-only blockers
    if allow_tags is not None and (tag in allow_tags or exp_id in allow_tags):
        txt = script_path.read_text(encoding="utf-8", errors="replace")
        has_cc8 = bool(re.search(r"get_device_capability|cc\[0\]\s*>=\s*8", txt))
        has_bf16_flag = "--bf16" in txt
        if has_bf16_flag and not has_cc8:
            return True, "explicitly allowed (wrapper will strip --bf16 for T4)"
        return True, "explicitly allowed via --only-ids"

    return False, "not in verified T4-safe whitelist (BF16-only or heavy L4 proxy training)"


def _dep_satisfied(
    dep_id: str,
    queue_by_id: Dict[str, Dict[str, Any]],
    *,
    hf_token: Optional[str],
    hf_cache: Dict[str, bool],
    check_hf: bool = True,
) -> Tuple[bool, str]:
    row = queue_by_id.get(dep_id)
    if row is not None and str(row.get("status", "")).upper() == "COMPLETED":
        return True, "queue:COMPLETED"
    if _local_outcome_exists(dep_id):
        return True, "local:OUTCOME.md"
    if check_hf:
        if dep_id in hf_cache:
            return hf_cache[dep_id], "hf:cached" if hf_cache[dep_id] else "hf:missing"
        ok = _hf_file_exists(f"evals/{dep_id}.json", token=hf_token)
        hf_cache[dep_id] = ok
        if ok:
            return True, "hf:eval_exists"
    return False, "pending"


def _tar_exclude(path: Path) -> bool:
    parts = set(path.parts)
    if "__pycache__" in parts or ".git" in parts or "target" in parts:
        return True
    if path.suffix in {".pyc", ".pyo"}:
        return True
    return False


def _build_and_upload_shared_bundle(
    bundle_path: Path,
    *,
    hf_token: str,
    skip_upload: bool = False,
) -> str:
    """Build lightweight shared tarball of `tmp-train/mvd/` (+ Rust crates) and upload to HF."""
    mvd_dir = REPO_ROOT / "tmp-train" / "mvd"
    if not mvd_dir.is_dir():
        raise FileNotFoundError(f"Missing required directory: {mvd_dir}")

    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_bundle = bundle_path.with_name(bundle_path.name + ".tmp")

    include_dirs: List[Tuple[Path, str]] = [
        (mvd_dir, "sheet-music-converter/tmp-train/mvd"),
    ]
    for rel in ("converter/assembly-serialize", "converter/eval-harness"):
        d = REPO_ROOT / rel
        if d.is_dir():
            include_dirs.append((d, f"sheet-music-converter/{rel}"))
    mscz_rs = REPO_ROOT.parent / "mscz-rs"
    if mscz_rs.is_dir():
        include_dirs.append((mscz_rs, "mscz-rs"))

    n_files = 0
    with tarfile.open(tmp_bundle, "w:gz") as tf:
        for src_dir, arc_prefix in include_dirs:
            for item in sorted(src_dir.rglob("*")):
                if not item.is_file() or _tar_exclude(item):
                    continue
                rel = item.relative_to(src_dir).as_posix()
                tf.add(item, arcname=f"{arc_prefix}/{rel}", recursive=False)
                n_files += 1

    os.replace(tmp_bundle, bundle_path)
    sha = hashlib.sha256(bundle_path.read_bytes()).hexdigest()[:12]
    size_mb = bundle_path.stat().st_size / (1024 * 1024)
    _log_console(
        f"Built Kaggle code bundle {bundle_path} ({n_files} files, {size_mb:.2f} MB, sha256={sha})"
    )

    if not skip_upload:
        from huggingface_hub import HfApi  # type: ignore

        api = HfApi(token=hf_token)
        api.upload_file(
            path_or_fileobj=str(bundle_path),
            path_in_repo=HF_BUNDLE_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            commit_message=f"kaggle-queue: update shared code bundle ({sha})",
        )
        _log_console(f"Uploaded shared code bundle to HF {HF_REPO_ID}:{HF_BUNDLE_PATH}")
    return sha


def _sanitize_script_for_t4(source: str) -> str:
    """Ensure T4-safe precision: never pass `--bf16` CLI flag on cc < 8.0."""
    # Replace literal `"--bf16"` CLI args with `"--fp16"` if a script invokes a training CLI
    # (verified T4-safe scripts already check `cc[0] >= 8` and have zero `--bf16` flags).
    sanitized = source.replace('"--bf16"', '"--fp16"').replace("'--bf16'", "'--fp16'")
    return sanitized


def _build_kaggle_wrapper_source(
    *,
    exp_id: str,
    script_source: str,
    hf_eval_target: str,
) -> str:
    """Build self-contained Kaggle kernel script wrapping the target experiment."""
    safe_src = _sanitize_script_for_t4(script_source)
    b64_script = base64.b64encode(safe_src.encode("utf-8")).decode("ascii")

    return f'''#!/usr/bin/env python3
"""Auto-generated Kaggle 2x T4 wrapper for {exp_id}."""
from __future__ import annotations

import base64
import glob
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

EXP_ID = {exp_id!r}
HF_REPO_ID = {HF_REPO_ID!r}
HF_BUNDLE_PATH = {HF_BUNDLE_PATH!r}
HF_EVAL_TARGET = {hf_eval_target!r}
EMBEDDED_SCRIPT_B64 = {b64_script!r}


def _read_hf_token() -> str:
    tok = os.environ.get("HF_TOKEN", "").strip()
    if tok:
        return tok
    for p in sorted(glob.glob("/kaggle/input/**/hf_token.txt", recursive=True)):
        try:
            val = Path(p).read_text(encoding="utf-8").strip()
            if val:
                return val
        except OSError:
            pass
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        val = (UserSecretsClient().get_secret("HF_TOKEN") or "").strip()
        if val:
            return val
    except Exception:
        pass
    raise RuntimeError(
        "HF_TOKEN not found in env, /kaggle/input/**/hf_token.txt, or kaggle_secrets"
    )


def _ensure_deps() -> None:
    needed = []
    for mod, pkg in [
        ("huggingface_hub", "huggingface_hub>=0.26"),
        ("scipy", "scipy"),
        ("tqdm", "tqdm"),
        ("PIL", "pillow"),
        ("numpy", "numpy"),
    ]:
        try:
            __import__(mod)
        except ImportError:
            needed.append(pkg)
    if needed:
        print(f"[kaggle:{{EXP_ID}}] Installing missing deps: {{needed}}", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *needed])


def _ensure_symlink(link_path: Path, target: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink() or link_path.exists():
        if link_path.is_symlink() and link_path.resolve() == target.resolve():
            return
        if link_path.is_symlink() or link_path.is_file():
            link_path.unlink()
        else:
            shutil.rmtree(link_path)
    link_path.symlink_to(target)


def main() -> int:
    hf_tok_file = Path("/tmp/.hf_token")
    try:
        # 1. Load HF token securely (NEVER print it!)
        hf_token = _read_hf_token()
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token

        # 2. Verify GPU & T4 precision guard
        try:
            import torch

            if torch.cuda.is_available():
                n_gpu = torch.cuda.device_count()
                name0 = torch.cuda.get_device_name(0)
                cc0 = torch.cuda.get_device_capability(0)
                print(
                    f"[kaggle:{{EXP_ID}}] CUDA devices={{n_gpu}} gpu0={{name0}} "
                    f"cc={{cc0[0]}}.{{cc0[1]}} torch={{torch.__version__}} "
                    f"(native_bf16={{cc0[0] >= 8}})",
                    flush=True,
                )
            else:
                print(f"[kaggle:{{EXP_ID}}] WARNING: CUDA not available!", flush=True)
        except Exception as exc:
            print(f"[kaggle:{{EXP_ID}}] GPU probe warning: {{exc}}", flush=True)

        # 3. Install lightweight missing deps
        _ensure_deps()

        # 4. Skip immediately if eval JSON already exists on HF
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi(token=hf_token)
        if api.file_exists(HF_REPO_ID, HF_EVAL_TARGET, repo_type="dataset"):
            print(
                f"[kaggle:{{EXP_ID}}] Eval {{HF_EVAL_TARGET}} already exists on HF; skipping.",
                flush=True,
            )
            return 0

        # 5. Download & extract shared code bundle into /tmp (keeps /kaggle/working clean)
        work_base = Path("/tmp/ov_work")
        cache_base = Path("/tmp/ov_cache")
        vol_base = Path("/tmp/ov_vol")
        work_base.mkdir(parents=True, exist_ok=True)
        cache_base.mkdir(parents=True, exist_ok=True)
        vol_base.mkdir(parents=True, exist_ok=True)

        print(f"[kaggle:{{EXP_ID}}] Downloading {{HF_BUNDLE_PATH}} from HF...", flush=True)
        bundle_local = Path(
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=HF_BUNDLE_PATH,
                repo_type="dataset",
                token=hf_token,
                local_dir="/tmp/ov_bundle_dl",
            )
        )
        with tarfile.open(bundle_local, "r:*") as tf:
            tf.extractall(path=work_base)

        repo_root = work_base / "sheet-music-converter"
        mvd_root = repo_root / "tmp-train" / "mvd"
        if not mvd_root.is_dir():
            raise FileNotFoundError(f"Unpacked bundle missing {{mvd_root}}")

        # 6. Wire all /content and /workspace symlinks expected by ov-* scripts
        _ensure_symlink(Path("/content/sheet-music-converter"), repo_root)
        _ensure_symlink(Path("/content/workspace/sheet-music-converter"), repo_root)
        _ensure_symlink(Path("/workspace/sheet-music-converter"), repo_root)
        _ensure_symlink(Path("/content/mvd"), mvd_root)
        _ensure_symlink(Path("/workspace/mvd"), mvd_root)
        _ensure_symlink(Path("/workspace/tmp-train/mvd"), mvd_root)
        _ensure_symlink(Path("/content/cache"), cache_base)
        _ensure_symlink(Path("/content/vol"), vol_base)

        mscz_root = work_base / "mscz-rs"
        if mscz_root.is_dir():
            _ensure_symlink(Path("/content/mscz-rs"), mscz_root)
            _ensure_symlink(Path("/workspace/mscz-rs"), mscz_root)
            _ensure_symlink(Path("/content/workspace/mscz-rs"), mscz_root)

        # 7. Materialize embedded experiment script inside repo_root/experiments/<exp_id>/
        # so scripts using Path(__file__).resolve().parents[2] resolve cleanly to repo_root.
        run_script = repo_root / "experiments" / EXP_ID / "run_exp.py"
        run_script.parent.mkdir(parents=True, exist_ok=True)
        run_script.write_bytes(base64.b64decode(EMBEDDED_SCRIPT_B64))

        env = os.environ.copy()
        py_path = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{{mvd_root}}:{{py_path}}" if py_path else str(mvd_root)
        env["PYTHONUNBUFFERED"] = "1"
        env["HF_TOKEN"] = hf_token

        print(f"[kaggle:{{EXP_ID}}] Launching {{run_script}} (cwd={{repo_root}})...", flush=True)
        proc = subprocess.run(
            [sys.executable, "-u", str(run_script)],
            cwd=str(repo_root),
            env=env,
            check=False,
        )
        print(f"[kaggle:{{EXP_ID}}] Finished rc={{proc.returncode}}", flush=True)
        return int(proc.returncode)
    finally:
        if hf_tok_file.exists():
            try:
                hf_tok_file.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _run_kaggle_cli(
    kaggle_exe: Path,
    args: Sequence[str],
    *,
    timeout_s: int = 120,
) -> Tuple[int, str]:
    cmd = [str(kaggle_exe), *args]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_s,
        check=False,
    )
    return int(proc.returncode), proc.stdout.strip()


def _parse_kaggle_status(output: str) -> str:
    low = output.lower()
    if "complete" in low and "error" not in low:
        return "COMPLETE"
    if "error" in low or "fail" in low:
        return "ERROR"
    if "cancel" in low:
        return "CANCELLED"
    if "running" in low:
        return "RUNNING"
    if "queued" in low:
        return "QUEUED"
    return "UNKNOWN"


def _decode_kaggle_log_file(raw_log_path: Path, dest_log_path: Path) -> None:
    """Decode Kaggle's JSON-array `<slug>.log` into readable plain text."""
    dest_log_path.parent.mkdir(parents=True, exist_ok=True)
    raw_text = raw_log_path.read_text(encoding="utf-8", errors="replace")
    try:
        entries = json.loads(raw_text)
        if isinstance(entries, list):
            lines: List[str] = []
            for item in entries:
                if isinstance(item, dict):
                    ts = item.get("time", "")
                    stream = item.get("stream_name", "stdout")
                    data = str(item.get("data", ""))
                    lines.append(f"[{ts}][{stream}] {data}")
                else:
                    lines.append(str(item))
            dest_log_path.write_text("".join(lines), encoding="utf-8", newline="\n")
            return
    except json.JSONDecodeError:
        pass
    dest_log_path.write_text(raw_text, encoding="utf-8", newline="\n")


def _run_one_kaggle_experiment(
    exp: Dict[str, Any],
    *,
    queue_path: Path,
    kaggle_exe: Path,
    kaggle_user: str,
    kernel_work_dir: Path,
    log_dir: Path,
    hf_token: str,
    poll_secs: int,
    max_wait_minutes: int,
) -> Dict[str, Any]:
    exp_id = str(exp["id"])
    script_path = REPO_ROOT / str(exp["script"])
    hf_eval_target = str(exp["hf_eval_target"])
    slug = _kaggle_slug_for(exp_id)
    kernel_ref = f"{kaggle_user}/{slug}"

    # 1. Auto-skip if already on HF
    if _hf_file_exists(hf_eval_target, token=hf_token):
        _log_console(f"[SKIP] {exp_id} already exists on HF ({hf_eval_target})")
        _update_experiment_record(
            queue_path,
            exp_id,
            {
                "status": "COMPLETED",
                "completed_at": _utc_now(),
                "skipped_existing_hf": True,
            },
        )
        return {"id": exp_id, "status": "COMPLETED", "skipped": True, "returncode": 0}

    # 2. Atomically claim in queue.json (refuses if Colab/another worker already took it)
    started_at = _utc_now()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{exp_id}.kaggle.log"
    claimed = _claim_experiment_atomic(
        queue_path,
        exp_id,
        {
            "status": "RUNNING",
            "provider": "kaggle",
            "gpu": "2xT4",
            "kernel_id": kernel_ref,
            "started_at": started_at,
            "log_path": str(log_path),
        },
    )
    if not claimed:
        _log_console(
            f"[SKIP-CLAIMED] {exp_id} is already RUNNING/COMPLETED in queue.json"
        )
        return {"id": exp_id, "status": "SKIPPED", "skipped": True, "returncode": 0}

    t0 = time.monotonic()
    kdir = kernel_work_dir / slug
    out_dir = kdir / "output"
    if kdir.exists():
        shutil.rmtree(kdir, ignore_errors=True)
    kdir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    _log_console(f"[START] {exp_id} -> Kaggle kernel {kernel_ref} (2xT4)")

    try:
        script_source = script_path.read_text(encoding="utf-8")
        wrapper_src = _build_kaggle_wrapper_source(
            exp_id=exp_id,
            script_source=script_source,
            hf_eval_target=hf_eval_target,
        )
        wrapper_path = kdir / "run_kaggle_wrapper.py"
        wrapper_path.write_text(wrapper_src, encoding="utf-8", newline="\n")

        # Kaggle derives the canonical kernel slug from `title`, NOT `id`.
        # Setting `title = slug` (already lowercase a-z0-9-) guarantees a 1:1 match
        # with zero slug-mismatch drift (e.g. avoids double `omr-omr-...` prefixes).
        title = slug
        meta = {
            "id": kernel_ref,
            "title": title,
            "code_file": "run_kaggle_wrapper.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_tpu": False,
            "enable_internet": True,
            "dataset_sources": [KAGGLE_HF_AUTH_DATASET],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
        (kdir / "kernel-metadata.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )

        # 3. Push kernel to Kaggle & verify canonical slug from push stdout
        # Kaggle returns rc=0 with "Kernel push error: Maximum batch GPU session count of 2 reached."
        # when both free-tier GPU slots are occupied; treat as transient [WAIT-SLOT] and retry.
        max_slot_retries = 360
        rc_push, out_push = -1, ""
        for slot_attempt in range(1, max_slot_retries + 1):
            rc_push, out_push = _run_kaggle_cli(kaggle_exe, ["kernels", "push", "-p", str(kdir)])
            if "maximum batch gpu session count" in out_push.lower():
                if slot_attempt == 1 or slot_attempt % 5 == 0:
                    _log_console(
                        f"[WAIT-SLOT] {exp_id} ({kernel_ref}) Kaggle GPU slots busy "
                        f"(attempt {slot_attempt}/{max_slot_retries}); retrying in 60s | {out_push}"
                    )
                if slot_attempt >= max_slot_retries:
                    raise RuntimeError(
                        f"kaggle kernels push timed out waiting for GPU slot after {max_slot_retries} attempts: {out_push}"
                    )
                time.sleep(60)
                continue
            _log_console(f"[PUSH] {exp_id} rc={rc_push}: {out_push}")
            break

        if rc_push != 0:
            raise RuntimeError(f"kaggle kernels push failed (rc={rc_push}): {out_push}")
        if "successfully pushed" not in out_push.lower():
            raise RuntimeError(
                f"kaggle kernels push returned rc=0 without push confirmation: {out_push}"
            )
        pushed_ref = _extract_pushed_kernel_ref(out_push)
        if pushed_ref and pushed_ref != kernel_ref:
            _log_console(
                f"[WARN] {exp_id} Kaggle resolved slug {pushed_ref!r} != requested {kernel_ref!r}; using {pushed_ref!r}"
            )
            kernel_ref = pushed_ref
            _update_experiment_record(
                queue_path,
                exp_id,
                {"kernel_id": kernel_ref},
            )

        # 4. Poll until terminal state
        deadline = time.monotonic() + max_wait_minutes * 60.0
        last_state = "QUEUED"
        poll_idx = 0
        consecutive_unknown = 0
        while True:
            time.sleep(max(5, int(poll_secs)))
            poll_idx += 1
            rc_st, out_st = _run_kaggle_cli(kaggle_exe, ["kernels", "status", kernel_ref])
            state = _parse_kaggle_status(out_st) if rc_st == 0 else "UNKNOWN"
            if state == "UNKNOWN":
                consecutive_unknown += 1
            else:
                consecutive_unknown = 0
            if state != last_state or poll_idx % 6 == 0:
                elapsed_m = (time.monotonic() - t0) / 60.0
                _log_console(
                    f"[POLL] {exp_id} ({kernel_ref}) state={state} ({elapsed_m:.1f}m) | {out_st}"
                )
                last_state = state

            if state in {"COMPLETE", "ERROR", "CANCELLED"}:
                break
            if consecutive_unknown >= 5:
                raise RuntimeError(
                    f"Kaggle status poll failed {consecutive_unknown}x consecutively for {kernel_ref}: {out_st}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Kaggle kernel {kernel_ref} exceeded max_wait_minutes={max_wait_minutes}"
                )

        # 5. Fetch & decode Kaggle logs
        rc_out, out_msg = _run_kaggle_cli(
            kaggle_exe, ["kernels", "output", kernel_ref, "-p", str(out_dir)]
        )
        if rc_out == 0:
            log_candidates = sorted(out_dir.glob("*.log"))
            if log_candidates:
                _decode_kaggle_log_file(log_candidates[0], log_path)
                _log_console(f"[LOG] {exp_id} decoded log -> {log_path}")
        else:
            _log_console(f"[WARN] {exp_id} kernels output failed (rc={rc_out}): {out_msg}")

        # 6. Verify HF eval artifact exists
        hf_ok = _hf_file_exists(hf_eval_target, token=hf_token)
        elapsed_sec = round(time.monotonic() - t0, 2)
        completed_at = _utc_now()
        if state == "COMPLETE" and hf_ok:
            _update_experiment_record(
                queue_path,
                exp_id,
                {
                    "status": "COMPLETED",
                    "provider": "kaggle",
                    "gpu": "2xT4",
                    "kernel_id": kernel_ref,
                    "completed_at": completed_at,
                    "elapsed_sec": elapsed_sec,
                    "returncode": 0,
                },
            )
            _log_console(
                f"[COMPLETED] {exp_id} on Kaggle ({kernel_ref}, elapsed={elapsed_sec:.1f}s, hf_verified=True)"
            )
            return {
                "id": exp_id,
                "status": "COMPLETED",
                "returncode": 0,
                "elapsed_sec": elapsed_sec,
            }

        reason = f"kaggle_state={state} hf_eval_exists={hf_ok}"
        _update_experiment_record(
            queue_path,
            exp_id,
            {
                "status": "FAILED",
                "provider": "kaggle",
                "gpu": "2xT4",
                "kernel_id": kernel_ref,
                "completed_at": completed_at,
                "elapsed_sec": elapsed_sec,
                "returncode": 1,
                "failure_reason": reason,
            },
        )
        _log_console(f"[FAILED] {exp_id} ({kernel_ref}): {reason}")
        return {
            "id": exp_id,
            "status": "FAILED",
            "returncode": 1,
            "elapsed_sec": elapsed_sec,
            "reason": reason,
        }
    except Exception as exc:
        elapsed_sec = round(time.monotonic() - t0, 2)
        _update_experiment_record(
            queue_path,
            exp_id,
            {
                "status": "FAILED",
                "provider": "kaggle",
                "gpu": "2xT4",
                "kernel_id": kernel_ref,
                "completed_at": _utc_now(),
                "elapsed_sec": elapsed_sec,
                "returncode": -1,
                "failure_reason": str(exc),
            },
        )
        _log_console(f"[ERROR] {exp_id} ({kernel_ref}): {exc}")
        return {
            "id": exp_id,
            "status": "FAILED",
            "returncode": -1,
            "elapsed_sec": elapsed_sec,
            "reason": str(exc),
        }


def _print_t4_schedule(
    queue: Sequence[Dict[str, Any]],
    *,
    allow_tags: Optional[Set[str]],
    order: str,
    max_parallel: int,
    hf_token: Optional[str],
    check_hf: bool,
) -> List[Dict[str, Any]]:
    queue_by_id = {str(e["id"]): e for e in queue}
    hf_cache: Dict[str, bool] = {}

    eligible: List[Tuple[Dict[str, Any], str, bool, str]] = []
    skipped_rows: List[Tuple[Dict[str, Any], str]] = []

    for e in queue:
        ok, reason = _is_t4_eligible(e, allow_tags=allow_tags)
        if not ok:
            skipped_rows.append((e, reason))
            continue
        deps = [str(d) for d in e.get("depends_on", [])]
        deps_ok = True
        dep_notes: List[str] = []
        for d in deps:
            dok, dwhy = _dep_satisfied(
                d, queue_by_id, hf_token=hf_token, hf_cache=hf_cache, check_hf=check_hf
            )
            if not dok:
                deps_ok = False
            dep_notes.append(f"{d}:{dwhy}")
        dep_str = ", ".join(dep_notes) if dep_notes else "none"
        eligible.append((e, reason, deps_ok, dep_str))

    if order == "reverse":
        eligible.sort(
            key=lambda x: (-int(x[0].get("priority", 0)), int(x[0].get("est_minutes", 999)), str(x[0]["id"]))
        )
    else:
        eligible.sort(
            key=lambda x: (int(x[0].get("priority", 999)), int(x[0].get("est_minutes", 999)), str(x[0]["id"]))
        )

    print(
        f"\nKaggle 2x T4 Cooperative DAG Schedule "
        f"(order={order}, max_parallel={max_parallel}, eligible={len(eligible)}/{len(queue)})"
    )
    print("=" * 124)
    print(
        f"{'Tag':<7} {'Pri':<4} {'Status':<10} {'DepsOK':<7} {'Min':>4} {'Kernel Slug':<36} {'Reason / Deps'}"
    )
    print("-" * 124)
    ready_list: List[Dict[str, Any]] = []
    for e, reason, deps_ok, dep_str in eligible:
        tag = _short_tag(str(e["id"]))
        st = str(e.get("status", "QUEUED")).upper()
        slug = _kaggle_slug_for(str(e["id"]))
        is_ready = (st == "QUEUED") and deps_ok
        if is_ready:
            ready_list.append(e)
        marker = "READY" if is_ready else ("WAIT" if st == "QUEUED" else st)
        print(
            f"{tag:<7} {e.get('priority', 0):<4} {marker:<10} {('YES' if deps_ok else 'NO'):<7} "
            f"{int(e.get('est_minutes', 0)):>4} {slug:<36} {reason} | deps={dep_str}"
        )
    print("=" * 124)
    print(
        f"T4-ELIGIBLE TOTAL: {len(eligible)} | READY NOW (QUEUED + DepsOK): {len(ready_list)} | "
        f"SKIPPED (L4/BF16-only): {len(skipped_rows)}\n"
    )
    return [x[0] for x in eligible]


def _parse_only_ids(arg: Optional[str]) -> Optional[Set[str]]:
    if not arg:
        return None
    out = {tok.strip() for tok in arg.split(",") if tok.strip()}
    return out or None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Parallel Kaggle 2x T4 DAG worker cooperating on queue.json."
    )
    parser.add_argument(
        "--queue",
        type=Path,
        default=DEFAULT_QUEUE_PATH,
        help="Path to shared overnight queue JSON.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="Max concurrent Kaggle GPU kernels (Kaggle allows up to 2; default: 1).",
    )
    parser.add_argument(
        "--order",
        choices=("reverse", "forward"),
        default="reverse",
        help="Pick order across eligible jobs: 'reverse' works from tail to avoid Colab at head (default: reverse).",
    )
    parser.add_argument(
        "--only-ids",
        type=str,
        default=None,
        help="Optional comma-separated short tags or full IDs to restrict to (e.g. 'ov-39,ov-37,ov-38,ov-21').",
    )
    parser.add_argument(
        "--kaggle-exe",
        type=Path,
        default=DEFAULT_KAGGLE_EXE,
        help="Path to kaggle.exe CLI binary.",
    )
    parser.add_argument(
        "--kaggle-user",
        type=str,
        default=KAGGLE_USERNAME,
        help="Kaggle username (default: gregoreesmaa).",
    )
    parser.add_argument(
        "--kernel-work-dir",
        type=Path,
        default=DEFAULT_KERNEL_WORK_DIR,
        help="Local staging directory for generated Kaggle kernels.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help="Directory for decoded Kaggle experiment logs.",
    )
    parser.add_argument(
        "--poll-secs",
        type=int,
        default=30,
        help="Seconds between `kaggle kernels status` polls (default: 30).",
    )
    parser.add_argument(
        "--max-wait-minutes",
        type=int,
        default=660,
        help="Per-kernel hard wait timeout in minutes (default: 660, under Kaggle 12h cap).",
    )
    parser.add_argument(
        "--skip-bundle-upload",
        action="store_true",
        help="Reuse existing HF code/ov_mvd_bundle.tar.gz instead of rebuilding/uploading.",
    )
    parser.add_argument(
        "--no-hf-check",
        action="store_true",
        help="In --dry-run, skip online HF existence checks for external dependencies.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect queue, verify T4 eligibility & dependencies, and print schedule without pushing kernels.",
    )
    args = parser.parse_args(argv)

    queue_path = args.queue if args.queue.is_absolute() else (REPO_ROOT / args.queue)
    allow_tags = _parse_only_ids(args.only_ids)
    hf_token = _resolve_hf_token()
    queue = _read_queue(queue_path)

    eligible_ordered = _print_t4_schedule(
        queue,
        allow_tags=allow_tags,
        order=args.order,
        max_parallel=args.max_parallel,
        hf_token=hf_token,
        check_hf=not args.no_hf_check,
    )

    if args.dry_run:
        print(
            "[DRY-RUN] Verified queue.json, T4-safe filter, and dependency status. "
            "No Kaggle kernels pushed."
        )
        return 0

    if not args.kaggle_exe.is_file():
        print(f"ERROR: kaggle.exe not found at {args.kaggle_exe}", file=sys.stderr)
        return 1

    if not hf_token:
        print(
            "ERROR: HF token not found in HF_TOKEN env var or ~/.cache/huggingface/token.",
            file=sys.stderr,
        )
        return 1

    if args.max_parallel > 2:
        _log_console(
            f"[WARN] Kaggle free tier allows at most 2 concurrent GPU kernels; clamping --max-parallel {args.max_parallel} -> 2."
        )
        args.max_parallel = 2

    temp_dir = Path(os.environ.get("TEMP") or tempfile.gettempdir())
    bundle_path = temp_dir / "ov_mvd_bundle_kaggle.tar.gz"
    if not args.skip_bundle_upload:
        _build_and_upload_shared_bundle(bundle_path, hf_token=hf_token, skip_upload=False)

    eligible_ids = [str(e["id"]) for e in eligible_ordered]
    attempted_ids: Set[str] = set()
    hf_dep_cache: Dict[str, bool] = {}
    failures = 0

    _log_console(
        f"=== Starting Kaggle 2x T4 Cooperative DAG Worker "
        f"(eligible={len(eligible_ids)}, max_parallel={args.max_parallel}, order={args.order}) ==="
    )

    with ThreadPoolExecutor(max_workers=max(1, int(args.max_parallel))) as pool:
        running: Dict[object, str] = {}
        while True:
            # Always re-read queue.json fresh so we see Colab's latest RUNNING/COMPLETED
            # updates AND any newly appended T4-safe follow-up entries!
            live_queue = _read_queue(queue_path)
            queue_by_id = {str(e["id"]): e for e in live_queue}

            live_eligible = [
                e for e in live_queue if _is_t4_eligible(e, allow_tags=allow_tags)[0]
            ]
            if args.order == "reverse":
                live_eligible.sort(
                    key=lambda x: (-int(x.get("priority", 0)), int(x.get("est_minutes", 999)), str(x["id"]))
                )
            else:
                live_eligible.sort(
                    key=lambda x: (int(x.get("priority", 999)), int(x.get("est_minutes", 999)), str(x["id"]))
                )

            waiting_deps = 0
            for row in live_eligible:
                exp_id = str(row["id"])
                if len(running) >= max(1, int(args.max_parallel)):
                    break
                if exp_id in attempted_ids or exp_id in running.values():
                    continue
                st = str(row.get("status", "QUEUED")).upper()
                if st in {"RUNNING", "COMPLETED", "FAILED", "BLOCKED"}:
                    continue

                deps = [str(d) for d in row.get("depends_on", [])]
                if not all(
                    _dep_satisfied(
                        d,
                        queue_by_id,
                        hf_token=hf_token,
                        hf_cache=hf_dep_cache,
                        check_hf=True,
                    )[0]
                    for d in deps
                ):
                    waiting_deps += 1
                    continue

                attempted_ids.add(exp_id)
                fut = pool.submit(
                    _run_one_kaggle_experiment,
                    dict(row),
                    queue_path=queue_path,
                    kaggle_exe=args.kaggle_exe,
                    kaggle_user=args.kaggle_user,
                    kernel_work_dir=args.kernel_work_dir,
                    log_dir=args.log_dir,
                    hf_token=hf_token,
                    poll_secs=args.poll_secs,
                    max_wait_minutes=args.max_wait_minutes,
                )
                running[fut] = exp_id

            if not running:
                if waiting_deps == 0:
                    break
                _log_console(
                    f"[WAIT-DEPS] {waiting_deps} T4-eligible job(s) waiting on predecessors; re-checking in 30s..."
                )
                time.sleep(30)
                continue

            done, _ = fut_wait(running.keys(), timeout=30.0, return_when=FIRST_COMPLETED)
            for fut in done:
                exp_id = running.pop(fut)
                res = fut.result()
                if res.get("status") == "FAILED":
                    failures += 1

    _log_console(f"Kaggle 2x T4 DAG worker finished. failures={failures}")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
