#!/usr/bin/env python3
"""Automated Google Colab L4 overnight wave runner for sheet-music-converter.

Reads `experiments/overnight_queue/queue.json`, skips any experiment whose
`hf_eval_target` already exists on Hugging Face Private Dataset Hub
(`gregoreesmaa/sheet-music-converter-omr`), builds one shared lightweight code
tarball of `tmp-train/mvd/` (plus Rust eval/serializer crates if present), and
runs experiments wave-by-wave with up to `--max-parallel` concurrent Colab L4
sessions.

Safety & Lifecycle Guarantees:
  - Always calls `colab.exe stop -s <session>` in a `finally:` block so no VM
    ever lingers idle burning compute units.
  - Passes `--timeout 0` to `colab.exe exec` to avoid the 30s silence timeout.
  - Uploads `~/.cache/huggingface/token` (or `HF_TOKEN` env var) to
    `/tmp/.hf_token` on the VM, sets `chmod 600`, deletes it in `finally:`, and
    NEVER prints the token.
  - Atomically updates `queue.json` (`tmp + os.replace`) under a threading lock.
  - Writes per-experiment logs to `D:\\tmp\\overnight-queue-logs\\<exp_id>.log`
    (outside git).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUEUE_PATH = REPO_ROOT / "experiments" / "overnight_queue" / "queue.json"
DEFAULT_LOG_DIR = Path(r"D:\tmp\overnight-queue-logs")
DEFAULT_COLAB_EXE = REPO_ROOT / ".venv" / "Scripts" / "colab.exe"

QUEUE_LOCK = threading.Lock()
PRINT_LOCK = threading.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log_console(msg: str) -> None:
    with PRINT_LOCK:
        print(f"[{_utc_now()}] {msg}", flush=True)


def _session_name_for(exp_id: str) -> str:
    m = re.search(r"(ov-\d+[a-z]*)", exp_id)
    if m:
        return m.group(1)
    slug = re.sub(r"[^a-z0-9-]+", "-", exp_id.lower()).strip("-")
    return slug[:24] or "ov-run"


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
    tmp = path.with_name(f"{path.name}.colab.{os.getpid()}.{threading.get_ident()}.tmp")
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
    with QUEUE_LOCK:
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
    """Atomically re-read queue.json and claim exp_id ONLY if not RUNNING/COMPLETED."""
    lock_path = queue_path.with_name(".queue.lock")
    with QUEUE_LOCK:
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
    """Re-read queue.json fresh under cross-process lock and update only exp_id."""
    lock_path = queue_path.with_name(".queue.lock")
    with QUEUE_LOCK:
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


def _local_outcome_exists(exp_id: str) -> bool:
    return (REPO_ROOT / "experiments" / exp_id / "OUTCOME.md").is_file()


def _resolve_hf_token_file() -> Tuple[Optional[Path], Optional[Path]]:
    """Return (token_file_path, temp_file_to_delete_or_none)."""
    default_tok = Path.home() / ".cache" / "huggingface" / "token"
    if default_tok.is_file() and default_tok.read_text(encoding="utf-8").strip():
        return default_tok, None
    env_tok = os.environ.get("HF_TOKEN", "").strip()
    if env_tok:
        fd, tmp_name = tempfile.mkstemp(prefix="ov_hf_tok_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
                f.write(env_tok)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
        p = Path(tmp_name)
        return p, p
    return None, None


def _hf_target_exists(hf_eval_target: str, token: Optional[str] = None) -> bool:
    try:
        from huggingface_hub import HfApi  # type: ignore

        api = HfApi(token=token) if token else HfApi()
        return bool(
            api.file_exists(
                HF_REPO_ID,
                hf_eval_target,
                repo_type="dataset",
            )
        )
    except Exception as exc:
        _log_console(f"[WARN] HF existence check failed for {hf_eval_target}: {exc}")
        return False


def _tar_exclude(path: Path) -> bool:
    parts = set(path.parts)
    if "__pycache__" in parts or ".git" in parts or "target" in parts:
        return True
    if path.suffix in {".pyc", ".pyo"}:
        return True
    return False


def _build_shared_bundle(bundle_path: Path) -> Path:
    """Build lightweight shared tarball of `tmp-train/mvd/` (plus Rust eval crates)."""
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
    size_mb = bundle_path.stat().st_size / (1024 * 1024)
    _log_console(
        f"Built shared bundle {bundle_path} ({n_files} files, {size_mb:.2f} MB)"
    )
    return bundle_path


def _build_bootstrap_source(exp_id: str) -> str:
    """Return a UTF-8 Python bootstrap script executed inside the Colab VM."""
    return f'''#!/usr/bin/env python3
"""Colab VM bootstrapper for {exp_id}."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

EXP_ID = {exp_id!r}
BUNDLE = Path("/content/ov_mvd_bundle.tar.gz")
HF_TOK = Path("/tmp/.hf_token")
RUN_SCRIPT = Path("/content/run_exp.py")


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


def _install_deps() -> None:
    pkgs = ["huggingface_hub", "scipy", "tqdm"]
    uv_bin = shutil.which("uv")
    if uv_bin:
        rc = subprocess.call([uv_bin, "pip", "install", "--system", "-q", *pkgs])
        if rc == 0:
            return
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])


def main() -> int:
    try:
        if HF_TOK.is_file():
            try:
                os.chmod(HF_TOK, 0o600)
            except OSError:
                pass

        if not BUNDLE.is_file():
            raise FileNotFoundError(f"Missing uploaded bundle: {{BUNDLE}}")

        with tarfile.open(BUNDLE, "r:*") as tf:
            tf.extractall(path="/content")

        repo_root = Path("/content/sheet-music-converter")
        mvd_root = repo_root / "tmp-train" / "mvd"
        if not mvd_root.is_dir():
            raise FileNotFoundError(f"Unpacked bundle missing {{mvd_root}}")

        # Expose mvd and repo root at every path expected across ov-13..42
        _ensure_symlink(Path("/content/mvd"), mvd_root)
        _ensure_symlink(Path("/workspace/mvd"), mvd_root)
        _ensure_symlink(Path("/workspace/tmp-train/mvd"), mvd_root)
        _ensure_symlink(Path("/workspace/sheet-music-converter"), repo_root)
        _ensure_symlink(Path("/content/workspace/sheet-music-converter"), repo_root)

        mscz_root = Path("/content/mscz-rs")
        if mscz_root.is_dir():
            _ensure_symlink(Path("/workspace/mscz-rs"), mscz_root)
            _ensure_symlink(Path("/content/workspace/mscz-rs"), mscz_root)

        _install_deps()

        env = os.environ.copy()
        py_path = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{{mvd_root}}:{{py_path}}" if py_path else str(mvd_root)
        env["PYTHONUNBUFFERED"] = "1"
        if str(mvd_root) not in sys.path:
            sys.path.insert(0, str(mvd_root))

        # Copy uploaded script into repo_root/experiments/<EXP_ID>/run_exp.py so
        # scripts using Path(__file__).resolve().parents[2] resolve to repo_root.
        exec_script = repo_root / "experiments" / EXP_ID / "run_exp.py"
        exec_script.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(RUN_SCRIPT, exec_script)

        print(f"[bootstrap:{{EXP_ID}}] Launching {{exec_script}} (mvd={{mvd_root}})", flush=True)
        proc = subprocess.run(
            [sys.executable, "-u", str(exec_script)],
            cwd=str(repo_root),
            env=env,
            check=False,
        )
        print(f"[bootstrap:{{EXP_ID}}] Finished rc={{proc.returncode}}", flush=True)
        return int(proc.returncode)
    finally:
        if HF_TOK.exists():
            try:
                HF_TOK.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _run_subprocess_logged(
    cmd: Sequence[str],
    log_handle: Any,
    *,
    label: str,
) -> int:
    log_handle.write(f"\n[{_utc_now()}] $ {' '.join(cmd)}\n")
    log_handle.flush()
    proc = subprocess.Popen(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        log_handle.write(line)
        log_handle.flush()
    rc = proc.wait()
    log_handle.write(f"[{_utc_now()}] [{label}] exit_code={rc}\n")
    log_handle.flush()
    return int(rc)


def _run_one_experiment(
    exp: Dict[str, Any],
    *,
    queue_path: Path,
    colab_exe: Path,
    gpu: str,
    bundle_path: Path,
    hf_token_file: Path,
    hf_token_str: Optional[str],
    log_dir: Path,
) -> Dict[str, Any]:
    exp_id = str(exp["id"])
    script_path = REPO_ROOT / str(exp["script"])
    hf_eval_target = str(exp["hf_eval_target"])
    session = _session_name_for(exp_id)

    if _hf_target_exists(hf_eval_target, token=hf_token_str):
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

    if not script_path.is_file():
        _log_console(f"[FAIL] {exp_id} missing script {script_path}")
        _update_experiment_record(
            queue_path,
            exp_id,
            {"status": "FAILED", "returncode": -2, "completed_at": _utc_now()},
        )
        return {"id": exp_id, "status": "FAILED", "returncode": -2}

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{exp_id}.log"
    started_at = _utc_now()
    t0 = time.monotonic()

    claimed = _claim_experiment_atomic(
        queue_path,
        exp_id,
        {
            "status": "RUNNING",
            "provider": "colab_l4",
            "session": session,
            "gpu": gpu,
            "started_at": started_at,
            "log_path": str(log_path),
        },
    )
    if not claimed:
        _log_console(
            f"[SKIP-CLAIMED] {exp_id} is already RUNNING/COMPLETED in queue.json"
        )
        return {"id": exp_id, "status": "SKIPPED", "skipped": True, "returncode": 0}
    _log_console(f"[START] {exp_id} (wave={exp['wave']}, session={session}, gpu={gpu})")

    bootstrap_tmp: Optional[Path] = None
    session_created = False
    rc = -1
    try:
        with open(log_path, "w", encoding="utf-8", newline="\n") as log_f:
            log_f.write(
                f"=== {exp_id} | wave={exp['wave']} | session={session} | gpu={gpu} | started={started_at} ===\n"
            )
            log_f.flush()

            rc_new = 1
            for attempt in range(1, 361):
                rc_new = _run_subprocess_logged(
                    [str(colab_exe), "new", "-s", session, "--gpu", gpu],
                    log_f,
                    label="colab-new",
                )
                if rc_new == 0:
                    break
                if attempt == 1 or attempt % 10 == 0:
                    _log_console(
                        f"[WAIT-SLOT] {exp_id} ({session}): Colab {gpu} busy (attempt {attempt}/360), retrying in 60s..."
                    )
                time.sleep(60)
            if rc_new != 0:
                rc = rc_new
                raise RuntimeError(f"colab new failed after retries (rc={rc_new})")
            session_created = True

            rc_up1 = _run_subprocess_logged(
                [
                    str(colab_exe),
                    "upload",
                    "-s",
                    session,
                    str(bundle_path),
                    "/content/ov_mvd_bundle.tar.gz",
                ],
                log_f,
                label="upload-bundle",
            )
            if rc_up1 != 0:
                rc = rc_up1
                raise RuntimeError(f"upload bundle failed (rc={rc_up1})")

            rc_up2 = _run_subprocess_logged(
                [
                    str(colab_exe),
                    "upload",
                    "-s",
                    session,
                    str(hf_token_file),
                    "/tmp/.hf_token",
                ],
                log_f,
                label="upload-hf-token",
            )
            if rc_up2 != 0:
                rc = rc_up2
                raise RuntimeError(f"upload hf token failed (rc={rc_up2})")

            rc_up3 = _run_subprocess_logged(
                [
                    str(colab_exe),
                    "upload",
                    "-s",
                    session,
                    str(script_path),
                    "/content/run_exp.py",
                ],
                log_f,
                label="upload-script",
            )
            if rc_up3 != 0:
                rc = rc_up3
                raise RuntimeError(f"upload script failed (rc={rc_up3})")

            fd, tmp_name = tempfile.mkstemp(prefix=f"ov_boot_{session}_", suffix=".py")
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as bf:
                bf.write(_build_bootstrap_source(exp_id))
            bootstrap_tmp = Path(tmp_name)

            rc = _run_subprocess_logged(
                [
                    str(colab_exe),
                    "exec",
                    "-s",
                    session,
                    "--timeout",
                    "0",
                    "-f",
                    str(bootstrap_tmp),
                ],
                log_f,
                label="colab-exec",
            )
    except Exception as exc:
        _log_console(f"[ERROR] {exp_id} ({session}): {exc}")
    finally:
        if bootstrap_tmp and bootstrap_tmp.exists():
            try:
                bootstrap_tmp.unlink()
            except OSError:
                pass
        if session_created:
            try:
                with open(log_path, "a", encoding="utf-8", newline="\n") as log_f:
                    _run_subprocess_logged(
                        [str(colab_exe), "stop", "-s", session],
                        log_f,
                        label="colab-stop",
                    )
            except Exception as stop_exc:
                _log_console(f"[WARN] Failed to stop session {session}: {stop_exc}")

    elapsed_sec = round(time.monotonic() - t0, 2)
    completed_at = _utc_now()
    status = "COMPLETED" if rc == 0 else "FAILED"
    _update_experiment_record(
        queue_path,
        exp_id,
        {
            "status": status,
            "started_at": started_at,
            "completed_at": completed_at,
            "elapsed_sec": elapsed_sec,
            "returncode": rc,
        },
    )
    _log_console(
        f"[{status}] {exp_id} (session={session}, rc={rc}, elapsed={elapsed_sec:.1f}s)"
    )
    return {
        "id": exp_id,
        "status": status,
        "returncode": rc,
        "elapsed_sec": elapsed_sec,
    }


def _parse_waves(waves_arg: Optional[str]) -> Optional[List[int]]:
    if not waves_arg:
        return None
    out: List[int] = []
    for tok in waves_arg.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return sorted(set(out))


def _print_schedule(selected: Sequence[Dict[str, Any]], gpu: str, max_parallel: int) -> None:
    by_wave: Dict[int, List[Dict[str, Any]]] = {}
    for e in selected:
        by_wave.setdefault(int(e["wave"]), []).append(e)

    print(
        f"\nOvernight Colab {gpu} Queue Schedule (max_parallel={max_parallel}, total={len(selected)} experiments)"
    )
    print("=" * 108)
    print(
        f"{'Wave':<5} {'Pri':<4} {'Session':<8} {'Min':>4} {'EstCU':>6} {'Status':<10} {'ID':<56} {'Script OK'}"
    )
    print("-" * 108)
    total_min = 0
    total_cu = 0.0
    for wave in sorted(by_wave):
        w_min = 0
        w_cu = 0.0
        for e in by_wave[wave]:
            sess = _session_name_for(e["id"])
            script_ok = (REPO_ROOT / e["script"]).is_file()
            w_min += int(e["est_minutes"])
            w_cu += float(e["est_cu"])
            print(
                f"W{wave:<4} {e.get('priority', 0):<4} {sess:<8} {e['est_minutes']:>4} {e['est_cu']:>6.2f} "
                f"{e.get('status', 'QUEUED'):<10} {e['id']:<56} {'YES' if script_ok else 'MISSING'}"
            )
        total_min += w_min
        total_cu += w_cu
        print(
            f"  --> Wave {wave} subtotal: {len(by_wave[wave])} exps | {w_min} min | {w_cu:.2f} CU"
        )
    print("=" * 108)
    print(
        f"TOTAL: {len(selected)} exps | {total_min} min (~{total_min / 60.0:.1f} GPU-hrs) | {total_cu:.2f} est CU\n"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Automated Google Colab L4 overnight wave runner."
    )
    parser.add_argument(
        "--queue",
        type=Path,
        default=DEFAULT_QUEUE_PATH,
        help="Path to master overnight queue JSON.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=3,
        help="Maximum concurrent Colab sessions per wave (default: 3).",
    )
    parser.add_argument(
        "--waves",
        type=str,
        default=None,
        help="Optional comma-separated wave filter, e.g. '4,5,6,7,8'.",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="L4",
        help="Colab GPU tier (default: L4).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help="Directory for per-experiment logs (default: D:\\tmp\\overnight-queue-logs).",
    )
    parser.add_argument(
        "--colab-exe",
        type=Path,
        default=DEFAULT_COLAB_EXE,
        help="Path to colab.exe CLI binary.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse queue, verify scripts, and print wave schedule without launching GPUs.",
    )
    args = parser.parse_args(argv)

    queue_path = args.queue if args.queue.is_absolute() else (REPO_ROOT / args.queue)
    queue = _read_queue(queue_path)
    wave_filter = _parse_waves(args.waves)

    selected = [
        e
        for e in queue
        if (wave_filter is None or int(e["wave"]) in wave_filter)
    ]
    selected.sort(key=lambda e: (int(e["wave"]), int(e["est_minutes"]), str(e["id"])))

    _print_schedule(selected, gpu=args.gpu, max_parallel=args.max_parallel)

    missing = [e["script"] for e in selected if not (REPO_ROOT / e["script"]).is_file()]
    if missing:
        print(f"ERROR: Missing {len(missing)} script(s): {missing}", file=sys.stderr)
        return 1

    if args.dry_run:
        print("[DRY-RUN] Verified queue and all script paths. No GPU sessions launched.")
        return 0

    if not args.colab_exe.is_file():
        print(f"ERROR: colab.exe not found at {args.colab_exe}", file=sys.stderr)
        return 1

    hf_token_file, temp_tok_to_del = _resolve_hf_token_file()
    if hf_token_file is None:
        print(
            "ERROR: HF token not found in ~/.cache/huggingface/token or HF_TOKEN env var.",
            file=sys.stderr,
        )
        return 1

    hf_token_str = hf_token_file.read_text(encoding="utf-8").strip()
    temp_dir = Path(os.environ.get("TEMP") or tempfile.gettempdir())
    bundle_path = temp_dir / "ov_mvd_bundle.tar.gz"
    _build_shared_bundle(bundle_path)

    from concurrent.futures import FIRST_COMPLETED, wait as fut_wait

    # Dynamic DAG scheduler: re-reads queue.json fresh on every tick under .queue.lock
    # so newly appended QUEUED follow-ups and cooperative Kaggle updates are picked up
    # automatically without restarting or clobbering queue.json.
    ext_dep_cache: Dict[str, bool] = {}
    attempted_ids: set[str] = set()

    def _dep_satisfied(dep_id: str, queue_by_id: Dict[str, Dict[str, Any]]) -> bool:
        row = queue_by_id.get(dep_id)
        if row is not None and str(row.get("status", "")).upper() == "COMPLETED":
            return True
        if _local_outcome_exists(dep_id):
            return True
        if row is not None:
            return False
        if ext_dep_cache.get(dep_id):
            return True
        ok = _hf_target_exists(f"evals/{dep_id}.json", token=hf_token_str)
        if ok:
            ext_dep_cache[dep_id] = True
        return ok

    init_queued = [
        e for e in selected if str(e.get("status", "QUEUED")).upper() == "QUEUED"
    ]
    init_done = [
        e for e in selected if str(e.get("status", "QUEUED")).upper() == "COMPLETED"
    ]
    _log_console(
        f"=== Starting Dynamic DAG Dependency-Chain Scheduler: {len(init_queued)} queued / {len(init_done)} already done (max_parallel={args.max_parallel}) ==="
    )
    failures = 0
    try:
        with ThreadPoolExecutor(max_workers=max(1, int(args.max_parallel))) as pool:
            running: Dict[object, Dict[str, Any]] = {}
            while True:
                # Re-read queue.json fresh on every tick so newly appended follow-ups
                # and Kaggle completions are picked up immediately.
                live_queue = _read_queue(queue_path)
                queue_by_id = {str(e["id"]): e for e in live_queue}
                failed_ids = {
                    str(e["id"])
                    for e in live_queue
                    if str(e.get("status", "")).upper() in {"FAILED", "BLOCKED"}
                }

                live_queued = [
                    e
                    for e in live_queue
                    if (wave_filter is None or int(e.get("wave", 0)) in wave_filter)
                    and str(e.get("status", "QUEUED")).upper() == "QUEUED"
                    and str(e["id"]) not in attempted_ids
                    and all(str(r["id"]) != str(e["id"]) for r in running.values())
                ]
                live_queued.sort(
                    key=lambda e: (
                        int(e.get("priority", 999)),
                        int(e.get("est_minutes", 999)),
                        str(e["id"]),
                    )
                )

                # 1. Fail-fast any QUEUED node whose in-queue predecessor FAILED/BLOCKED
                actionable: List[Dict[str, Any]] = []
                for exp in live_queued:
                    deps = [str(d) for d in exp.get("depends_on", [])]
                    bad_deps = [d for d in deps if d in failed_ids]
                    if bad_deps:
                        _log_console(
                            f"[BLOCKED] {exp['id']}: predecessor(s) failed: {bad_deps}"
                        )
                        _update_experiment_record(
                            queue_path,
                            str(exp["id"]),
                            {"status": "BLOCKED", "blocked_by": bad_deps},
                        )
                        failed_ids.add(str(exp["id"]))
                        failures += 1
                    else:
                        actionable.append(exp)

                # 2. Launch any ready experiment whose predecessors are all finished
                launched_any = False
                waiting_deps = 0
                for exp in actionable:
                    if len(running) >= max(1, int(args.max_parallel)):
                        break
                    deps = [str(d) for d in exp.get("depends_on", [])]
                    if all(_dep_satisfied(d, queue_by_id) for d in deps):
                        exp_id = str(exp["id"])
                        attempted_ids.add(exp_id)
                        fut = pool.submit(
                            _run_one_experiment,
                            dict(exp),
                            queue_path=queue_path,
                            colab_exe=args.colab_exe,
                            gpu=args.gpu,
                            bundle_path=bundle_path,
                            hf_token_file=hf_token_file,
                            hf_token_str=hf_token_str,
                            log_dir=args.log_dir,
                        )
                        running[fut] = exp
                        launched_any = True
                    else:
                        waiting_deps += 1

                if not running:
                    if waiting_deps == 0 and not actionable:
                        break
                    if not launched_any:
                        _log_console(
                            f"[WAIT-DEPS] {waiting_deps or len(actionable)} experiment(s) waiting on predecessor(s); re-checking in 30s..."
                        )
                        time.sleep(30)
                    continue

                done, _ = fut_wait(
                    running.keys(), timeout=30.0, return_when=FIRST_COMPLETED
                )
                for fut in done:
                    exp = running.pop(fut)
                    res = fut.result()
                    if res.get("status") not in {"COMPLETED", "SKIPPED"}:
                        failures += 1
    finally:
        if temp_tok_to_del and temp_tok_to_del.exists():
            try:
                temp_tok_to_del.unlink()
            except OSError:
                pass

    _log_console(f"Overnight queue finished. failures={failures}")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
