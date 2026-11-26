#!/usr/bin/env python3
"""Cooperative Modal GPU DAG queue worker for `experiments/overnight_queue/queue.json`.

Runs `QUEUED` experiments in parallel alongside `run_queue_colab.py` and
`run_queue_kaggle.py` using the exact same atomic `.queue.lock` + reload-before-write
claiming protocol so multiple cloud providers never collide on the same experiment.

Key Features:
  - Uses `experiments/overnight_queue/modal_runner.py` with `modal run --detach`
    + `fn.spawn(...)` so local network hiccups never cancel running GPU containers.
  - Streams live logs via `modal app logs <app_id> -f --function-call <call_id>` into
    `D:\\tmp\\overnight-queue-logs\\<exp_id>.modal.log` (with container-returned log
    fallback if log streaming drops).
  - Enforces `--budget-hours` (default 6.0 GPU-hours) and `--max-jobs` safety caps
    so the `$30/mo` Modal free tier is never unexpectedly exhausted.
  - Auto-skips any experiment whose `hf_eval_target` already exists on Hugging Face
    Private Dataset Hub (`gregoreesmaa/sheet-music-converter-omr`), and verifies
    `hf_eval_target` exists after completion.
  - NEVER prints or echoes `HF_TOKEN`, Modal tokens, or any secret values.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait as fut_wait
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

HF_REPO_ID = "gregoreesmaa/sheet-music-converter-omr"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUEUE_PATH = REPO_ROOT / "experiments" / "overnight_queue" / "queue.json"
DEFAULT_LOG_DIR = Path(r"D:\tmp\overnight-queue-logs")
DEFAULT_MODAL_EXE = REPO_ROOT / ".venv" / "Scripts" / "modal.exe"
DEFAULT_MODAL_RUNNER = REPO_ROOT / "experiments" / "overnight_queue" / "modal_runner.py"

# Approximate Modal hourly rates ($/hr) for dry-run cost visibility
MODAL_GPU_RATES_USD: Dict[str, float] = {
    "T4": 0.59,
    "L4": 0.80,
    "A10G": 1.10,
    "A100-40GB": 2.10,
    "A100-80GB": 2.50,
    "H100": 3.95,
}

PRINT_LOCK = threading.Lock()
THREAD_QUEUE_LOCK = threading.Lock()
BUDGET_LOCK = threading.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log_console(msg: str) -> None:
    with PRINT_LOCK:
        print(f"[{_utc_now()}] {msg}", flush=True)


def _short_tag(exp_id: str) -> str:
    m = re.search(r"(ov-\d+[a-z]*|i13-\d+[a-z]*|exp\d+[a-z]*)", exp_id)
    if m:
        return m.group(1)
    slug = re.sub(r"[^a-z0-9-]+", "-", exp_id.lower()).strip("-")
    return slug[:24] or "modal-run"


def _provider_name_for(gpu: str) -> str:
    clean = re.sub(r"[^a-z0-9]+", "_", gpu.lower()).strip("_")
    return f"modal_{clean or 'gpu'}"


# ---------------------------------------------------------------------------
# Cross-process `.queue.lock` + atomic `queue.json` protocol
# (identical semantics to run_queue_colab.py and run_queue_kaggle.py)
# ---------------------------------------------------------------------------


def _acquire_file_lock(lock_path: Path, timeout_s: float = 180.0) -> int:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
            os.write(fd, f"pid={os.getpid()} ts={_utc_now()}\n".encode("utf-8"))
            return fd
        except FileExistsError:
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
    tmp = path.with_name(f"{path.name}.modal.{os.getpid()}.{threading.get_ident()}.tmp")
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
    """Atomically re-read queue.json and claim exp_id ONLY if not RUNNING/COMPLETED."""
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
    """Re-read queue.json fresh under cross-process lock and update only exp_id."""
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


# ---------------------------------------------------------------------------
# Hugging Face & dependency helpers
# ---------------------------------------------------------------------------


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
    if row is not None:
        return False, f"queue:{row.get('status', 'QUEUED')}"
    if check_hf:
        if dep_id in hf_cache:
            return hf_cache[dep_id], "hf:cached" if hf_cache[dep_id] else "hf:missing"
        ok = _hf_file_exists(f"evals/{dep_id}.json", token=hf_token)
        hf_cache[dep_id] = ok
        if ok:
            return True, "hf:eval_exists"
    return False, "pending"


# ---------------------------------------------------------------------------
# Modal execution + log streaming
# ---------------------------------------------------------------------------


def _spawn_modal_experiment(
    *,
    modal_exe: Path,
    runner_path: Path,
    exp_id: str,
    script_path: Path,
    hf_eval_target: str,
    gpu: str,
    timeout_secs: int,
    log_path: Path,
) -> Tuple[str, str]:
    """Launch `modal run --detach` and return (app_id, call_id)."""
    cmd = [
        str(modal_exe),
        "run",
        "--detach",
        str(runner_path),
        "--exp-id",
        exp_id,
        "--script-path",
        str(script_path),
        "--hf-eval-target",
        hf_eval_target,
        "--gpu",
        gpu,
        "--timeout-secs",
        str(int(timeout_secs)),
    ]
    env = os.environ.copy()
    env["MODAL_QUEUE_GPU"] = gpu
    env["MODAL_QUEUE_TIMEOUT"] = str(int(timeout_secs))
    env["PYTHONUNBUFFERED"] = "1"

    app_id: Optional[str] = None
    call_id: Optional[str] = None
    with open(log_path, "a", encoding="utf-8", newline="\n") as log_f:
        log_f.write(f"[{_utc_now()}] $ {' '.join(cmd)}\n")
        log_f.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
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
            log_f.write(line)
            log_f.flush()
            m = re.search(r"MODAL_SPAWN_JSON=(\{.*\})", line)
            if m:
                try:
                    info = json.loads(m.group(1))
                    app_id = str(info.get("app_id") or "") or app_id
                    call_id = str(info.get("call_id") or "") or call_id
                except json.JSONDecodeError:
                    pass
        rc = proc.wait()
        log_f.write(f"[{_utc_now()}] [modal-spawn] exit_code={rc} app_id={app_id} call_id={call_id}\n")
        log_f.flush()

    if rc != 0 or not call_id:
        raise RuntimeError(
            f"modal run --detach failed for {exp_id} (rc={rc}, app_id={app_id}, call_id={call_id})"
        )
    return app_id or "", call_id


def _stream_modal_logs(
    *,
    modal_exe: Path,
    app_id: str,
    call_id: str,
    log_path: Path,
    stop_event: threading.Event,
) -> int:
    """Follow Modal app logs in a background thread and append them to log_path."""
    if not app_id:
        return 0
    cmd = [str(modal_exe), "app", "logs", app_id, "-f"]
    if call_id:
        cmd.extend(["--function-call", call_id])
    lines_written = 0
    proc: Optional[subprocess.Popen[str]] = None
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        with open(log_path, "a", encoding="utf-8", newline="\n") as log_f:
            for line in proc.stdout:
                log_f.write(line)
                log_f.flush()
                lines_written += 1
                if stop_event.is_set():
                    break
    except Exception:
        pass
    finally:
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
    return lines_written


def _run_one_modal_experiment(
    exp: Dict[str, Any],
    *,
    queue_path: Path,
    modal_exe: Path,
    runner_path: Path,
    gpu: str,
    timeout_secs: int,
    poll_secs: int,
    hf_token: Optional[str],
    log_dir: Path,
) -> Dict[str, Any]:
    exp_id = str(exp["id"])
    script_path = REPO_ROOT / str(exp["script"])
    hf_eval_target = str(exp.get("hf_eval_target", ""))
    provider = _provider_name_for(gpu)

    # 1. Auto-skip if already on HF
    if hf_eval_target and _hf_file_exists(hf_eval_target, token=hf_token):
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
        return {
            "id": exp_id,
            "status": "COMPLETED",
            "skipped": True,
            "returncode": 0,
            "elapsed_sec": 0.0,
        }

    if not script_path.is_file():
        _log_console(f"[FAIL] {exp_id} missing script {script_path}")
        _update_experiment_record(
            queue_path,
            exp_id,
            {"status": "FAILED", "returncode": -2, "completed_at": _utc_now()},
        )
        return {
            "id": exp_id,
            "status": "FAILED",
            "returncode": -2,
            "elapsed_sec": 0.0,
        }

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{exp_id}.modal.log"
    started_at = _utc_now()
    t0 = time.monotonic()

    # 2. Atomically claim in queue.json under .queue.lock
    claimed = _claim_experiment_atomic(
        queue_path,
        exp_id,
        {
            "status": "RUNNING",
            "provider": provider,
            "gpu": gpu,
            "started_at": started_at,
            "log_path": str(log_path),
        },
    )
    if not claimed:
        _log_console(f"[SKIP-CLAIMED] {exp_id} is already RUNNING/COMPLETED in queue.json")
        return {
            "id": exp_id,
            "status": "SKIPPED",
            "skipped": True,
            "returncode": 0,
            "elapsed_sec": 0.0,
        }

    _log_console(f"[START] {exp_id} -> Modal {gpu} (wave={exp.get('wave')}, pri={exp.get('priority')})")

    with open(log_path, "w", encoding="utf-8", newline="\n") as log_f:
        log_f.write(
            f"=== {exp_id} | provider={provider} | gpu={gpu} | wave={exp.get('wave')} | started={started_at} ===\n"
        )
        log_f.flush()

    app_id = ""
    call_id = ""
    stop_stream = threading.Event()
    stream_thread: Optional[threading.Thread] = None
    stream_lines = [0]

    try:
        # 3. Spawn detached Modal function call
        app_id, call_id = _spawn_modal_experiment(
            modal_exe=modal_exe,
            runner_path=runner_path,
            exp_id=exp_id,
            script_path=script_path,
            hf_eval_target=hf_eval_target,
            gpu=gpu,
            timeout_secs=timeout_secs,
            log_path=log_path,
        )
        _update_experiment_record(
            queue_path,
            exp_id,
            {
                "modal_app_id": app_id,
                "modal_call_id": call_id,
            },
        )
        _log_console(f"[SPAWNED] {exp_id} app_id={app_id} call_id={call_id}")

        # 4. Start live log streamer in background thread
        def _stream_worker() -> None:
            stream_lines[0] = _stream_modal_logs(
                modal_exe=modal_exe,
                app_id=app_id,
                call_id=call_id,
                log_path=log_path,
                stop_event=stop_stream,
            )

        stream_thread = threading.Thread(target=_stream_worker, daemon=True)
        stream_thread.start()

        # 5. Poll modal.FunctionCall until completion or hard timeout
        import modal

        fc = modal.FunctionCall.from_id(call_id)
        deadline = time.monotonic() + float(timeout_secs) + 600.0
        result: Optional[Dict[str, Any]] = None
        poll_idx = 0
        while True:
            try:
                result = fc.get(timeout=float(max(5, poll_secs)))
                break
            except TimeoutError:
                poll_idx += 1
                if poll_idx % 10 == 0:
                    elapsed_m = (time.monotonic() - t0) / 60.0
                    _log_console(
                        f"[RUNNING] {exp_id} ({call_id}, {elapsed_m:.1f}m elapsed, streamed_lines={stream_lines[0]})"
                    )
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Modal call {call_id} exceeded hard wait timeout ({timeout_secs}s + 600s)"
                    )

        # Give log streamer a moment to flush trailing lines, then stop it
        time.sleep(2.0)
        stop_stream.set()
        if stream_thread is not None:
            stream_thread.join(timeout=5.0)

        # If live streaming missed output, append the container-captured log_tail
        log_tail = str((result or {}).get("log_tail") or "")
        if log_tail and stream_lines[0] < 5:
            with open(log_path, "a", encoding="utf-8", newline="\n") as log_f:
                log_f.write("\n--- [container captured log fallback] ---\n")
                log_f.write(log_tail)
                if not log_tail.endswith("\n"):
                    log_f.write("\n")

        rc = int((result or {}).get("returncode", -1))
        hf_ok = bool((result or {}).get("hf_eval_exists", False))
        if hf_eval_target and not hf_ok:
            hf_ok = _hf_file_exists(hf_eval_target, token=hf_token)

        elapsed_sec = round(time.monotonic() - t0, 2)
        completed_at = _utc_now()
        ok = (rc == 0) and (hf_ok or not hf_eval_target)
        status = "COMPLETED" if ok else "FAILED"

        updates: Dict[str, Any] = {
            "status": status,
            "provider": provider,
            "gpu": gpu,
            "modal_app_id": app_id,
            "modal_call_id": call_id,
            "started_at": started_at,
            "completed_at": completed_at,
            "elapsed_sec": elapsed_sec,
            "returncode": rc,
            "hf_eval_exists": hf_ok,
        }
        if not ok:
            updates["failure_reason"] = f"modal_rc={rc} hf_eval_exists={hf_ok}"

        _update_experiment_record(queue_path, exp_id, updates)
        _log_console(
            f"[{status}] {exp_id} on Modal {gpu} (rc={rc}, hf_ok={hf_ok}, elapsed={elapsed_sec:.1f}s)"
        )
        return {
            "id": exp_id,
            "status": status,
            "returncode": rc,
            "elapsed_sec": elapsed_sec,
        }
    except Exception as exc:
        stop_stream.set()
        if stream_thread is not None:
            stream_thread.join(timeout=3.0)
        elapsed_sec = round(time.monotonic() - t0, 2)
        with open(log_path, "a", encoding="utf-8", newline="\n") as log_f:
            log_f.write(f"\n[{_utc_now()}] [ERROR] {exc}\n")
        _update_experiment_record(
            queue_path,
            exp_id,
            {
                "status": "FAILED",
                "provider": provider,
                "gpu": gpu,
                "modal_app_id": app_id,
                "modal_call_id": call_id,
                "started_at": started_at,
                "completed_at": _utc_now(),
                "elapsed_sec": elapsed_sec,
                "returncode": -1,
                "failure_reason": str(exc),
            },
        )
        _log_console(f"[ERROR] {exp_id} ({app_id}/{call_id}): {exc}")
        return {
            "id": exp_id,
            "status": "FAILED",
            "returncode": -1,
            "elapsed_sec": elapsed_sec,
            "reason": str(exc),
        }


# ---------------------------------------------------------------------------
# Schedule display & CLI
# ---------------------------------------------------------------------------


def _parse_waves(waves_arg: Optional[str]) -> Optional[Set[int]]:
    if not waves_arg:
        return None
    out = {int(tok.strip()) for tok in waves_arg.split(",") if tok.strip()}
    return out or None


def _parse_only_ids(arg: Optional[str]) -> Optional[Set[str]]:
    if not arg:
        return None
    out = {tok.strip() for tok in arg.split(",") if tok.strip()}
    return out or None


def _matches_filters(
    exp: Dict[str, Any],
    *,
    wave_filter: Optional[Set[int]],
    allow_ids: Optional[Set[str]],
) -> bool:
    if wave_filter is not None and int(exp.get("wave", 0)) not in wave_filter:
        return False
    if allow_ids is not None:
        exp_id = str(exp.get("id", ""))
        tag = _short_tag(exp_id)
        if exp_id not in allow_ids and tag not in allow_ids:
            return False
    return True


def _sort_experiments(exps: List[Dict[str, Any]], order: str) -> List[Dict[str, Any]]:
    if order == "reverse":
        return sorted(
            exps,
            key=lambda e: (-int(e.get("priority", 0)), int(e.get("est_minutes", 999)), str(e["id"])),
        )
    return sorted(
        exps,
        key=lambda e: (int(e.get("priority", 999)), int(e.get("est_minutes", 999)), str(e["id"])),
    )


def _print_schedule(
    queue: Sequence[Dict[str, Any]],
    *,
    gpu: str,
    max_parallel: int,
    budget_hours: float,
    max_jobs: int,
    order: str,
    wave_filter: Optional[Set[int]],
    allow_ids: Optional[Set[str]],
    hf_token: Optional[str],
    check_hf: bool,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    queue_by_id = {str(e["id"]): e for e in queue}
    hf_cache: Dict[str, bool] = {}

    filtered = [
        e for e in queue if _matches_filters(e, wave_filter=wave_filter, allow_ids=allow_ids)
    ]
    ordered = _sort_experiments(filtered, order)

    rate = MODAL_GPU_RATES_USD.get(gpu.upper(), 0.80)
    print(
        f"\nModal {gpu} Cooperative DAG Queue Schedule "
        f"(order={order}, max_parallel={max_parallel}, budget_hours={budget_hours:.2f}h "
        f"[~${budget_hours * rate:.2f}], max_jobs={max_jobs or 'unlimited'})"
    )
    print("=" * 124)
    print(
        f"{'Tag':<8} {'Wave':<5} {'Pri':<4} {'State':<10} {'DepsOK':<7} {'Min':>4} {'Script':<7} {'ID'}"
    )
    print("-" * 124)

    ready_exps: List[Dict[str, Any]] = []
    missing_scripts: List[str] = []
    counts: Dict[str, int] = {"READY": 0, "WAIT_DEPS": 0, "RUNNING": 0, "COMPLETED": 0, "FAILED": 0, "BLOCKED": 0}

    for e in ordered:
        exp_id = str(e["id"])
        tag = _short_tag(exp_id)
        st = str(e.get("status", "QUEUED")).upper()
        script_ok = (REPO_ROOT / str(e.get("script", ""))).is_file()
        if not script_ok and st == "QUEUED":
            missing_scripts.append(str(e.get("script", "")))

        deps = [str(d) for d in e.get("depends_on", [])]
        deps_ok = True
        for d in deps:
            dok, _ = _dep_satisfied(
                d, queue_by_id, hf_token=hf_token, hf_cache=hf_cache, check_hf=check_hf
            )
            if not dok:
                deps_ok = False
                break

        if st == "QUEUED":
            marker = "READY" if deps_ok else "WAIT_DEPS"
            if deps_ok and script_ok:
                ready_exps.append(e)
        else:
            marker = st
        counts[marker] = counts.get(marker, 0) + 1

        if st == "QUEUED" or allow_ids is not None:
            print(
                f"{tag:<8} W{int(e.get('wave', 0)):<4} {int(e.get('priority', 0)):<4} "
                f"{marker:<10} {('YES' if deps_ok else 'NO'):<7} "
                f"{int(e.get('est_minutes', 0)):>4} {('OK' if script_ok else 'MISSING'):<7} {exp_id}"
            )

    ready_min = sum(int(e.get("est_minutes", 30)) for e in ready_exps)
    print("=" * 124)
    print(
        f"SUMMARY: READY={counts['READY']} ({ready_min} est min / {ready_min / 60.0:.1f}h) | "
        f"WAIT_DEPS={counts['WAIT_DEPS']} | RUNNING={counts['RUNNING']} | "
        f"COMPLETED={counts['COMPLETED']} | FAILED={counts['FAILED']} | BLOCKED={counts['BLOCKED']}\n"
    )
    return ready_exps, missing_scripts


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cooperative Modal GPU DAG queue worker for experiments/overnight_queue/queue.json."
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
        default=2,
        help="Maximum concurrent Modal GPU jobs (default: 2).",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="L4",
        help="Modal GPU tier, e.g. L4, A10G, A100-40GB (default: L4).",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=0,
        help="Maximum number of jobs to launch before stopping (0 = unlimited).",
    )
    parser.add_argument(
        "--budget-hours",
        type=float,
        default=6.0,
        help="Safety cap on cumulative GPU-hours for this worker session (default: 6.0).",
    )
    parser.add_argument(
        "--order",
        choices=("forward", "reverse"),
        default="forward",
        help="Pick order across ready QUEUED jobs (default: forward).",
    )
    parser.add_argument(
        "--waves",
        type=str,
        default=None,
        help="Optional comma-separated wave filter, e.g. '5,6,7,8,9'.",
    )
    parser.add_argument(
        "--only-ids",
        type=str,
        default=None,
        help="Optional comma-separated short tags or full IDs (e.g. 'i13-01,i13-02').",
    )
    parser.add_argument(
        "--timeout-secs",
        type=int,
        default=7200,
        help="Per-experiment Modal container timeout in seconds (default: 7200 = 2h).",
    )
    parser.add_argument(
        "--poll-secs",
        type=int,
        default=20,
        help="Seconds between Modal FunctionCall.get() poll intervals (default: 20).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help="Directory for per-experiment Modal logs (default: D:\\tmp\\overnight-queue-logs).",
    )
    parser.add_argument(
        "--modal-exe",
        type=Path,
        default=DEFAULT_MODAL_EXE,
        help="Path to modal.exe CLI binary.",
    )
    parser.add_argument(
        "--runner",
        type=Path,
        default=DEFAULT_MODAL_RUNNER,
        help="Path to Modal App module (default: experiments/overnight_queue/modal_runner.py).",
    )
    parser.add_argument(
        "--no-hf-check",
        action="store_true",
        help="In --dry-run, skip online HF existence checks for external dependencies.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect queue, verify scripts & dependencies, and print schedule without mutating queue.json or launching Modal GPUs.",
    )
    args = parser.parse_args(argv)

    queue_path = args.queue if args.queue.is_absolute() else (REPO_ROOT / args.queue)
    runner_path = args.runner if args.runner.is_absolute() else (REPO_ROOT / args.runner)
    wave_filter = _parse_waves(args.waves)
    allow_ids = _parse_only_ids(args.only_ids)
    hf_token = _resolve_hf_token()

    # In --dry-run, read queue.json directly without creating/touching lockfile
    queue = _load_queue_unlocked(queue_path) if args.dry_run else _read_queue(queue_path)
    ready_exps, missing_scripts = _print_schedule(
        queue,
        gpu=args.gpu,
        max_parallel=args.max_parallel,
        budget_hours=args.budget_hours,
        max_jobs=args.max_jobs,
        order=args.order,
        wave_filter=wave_filter,
        allow_ids=allow_ids,
        hf_token=hf_token,
        check_hf=not args.no_hf_check,
    )

    if missing_scripts:
        print(f"WARN: {len(missing_scripts)} QUEUED script(s) missing: {missing_scripts}", file=sys.stderr)

    if args.dry_run:
        print(
            f"[DRY-RUN] Verified queue.json ({len(ready_exps)} ready QUEUED jobs). "
            "No Modal GPUs launched and queue.json was NOT modified."
        )
        return 0

    if not args.modal_exe.is_file():
        print(f"ERROR: modal.exe not found at {args.modal_exe}", file=sys.stderr)
        return 1
    if not runner_path.is_file():
        print(f"ERROR: modal_runner.py not found at {runner_path}", file=sys.stderr)
        return 1

    attempted_ids: Set[str] = set()
    hf_dep_cache: Dict[str, bool] = {}
    jobs_launched = 0
    used_gpu_hours = 0.0
    reserved_est_hours: Dict[object, float] = {}
    failures = 0
    budget_exhausted = False

    _log_console(
        f"=== Starting Cooperative Modal {args.gpu} DAG Worker "
        f"(max_parallel={args.max_parallel}, budget_hours={args.budget_hours:.2f}h, "
        f"max_jobs={args.max_jobs or 'unlimited'}, order={args.order}) ==="
    )

    with ThreadPoolExecutor(max_workers=max(1, int(args.max_parallel))) as pool:
        running: Dict[object, Dict[str, Any]] = {}
        while True:
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
                if _matches_filters(e, wave_filter=wave_filter, allow_ids=allow_ids)
                and str(e.get("status", "QUEUED")).upper() == "QUEUED"
                and str(e["id"]) not in attempted_ids
                and all(str(r["id"]) != str(e["id"]) for r in running.values())
            ]
            live_queued = _sort_experiments(live_queued, args.order)

            actionable: List[Dict[str, Any]] = []
            for exp in live_queued:
                deps = [str(d) for d in exp.get("depends_on", [])]
                bad_deps = [d for d in deps if d in failed_ids]
                if bad_deps:
                    _log_console(f"[BLOCKED] {exp['id']}: predecessor(s) failed: {bad_deps}")
                    _update_experiment_record(
                        queue_path,
                        str(exp["id"]),
                        {"status": "BLOCKED", "blocked_by": bad_deps},
                    )
                    failed_ids.add(str(exp["id"]))
                    failures += 1
                else:
                    actionable.append(exp)

            waiting_deps = 0
            for exp in actionable:
                if len(running) >= max(1, int(args.max_parallel)):
                    break
                if args.max_jobs > 0 and jobs_launched >= args.max_jobs:
                    break

                est_h = max(0.1, float(exp.get("est_minutes", 30)) / 60.0)
                with BUDGET_LOCK:
                    committed_h = used_gpu_hours + sum(reserved_est_hours.values())
                if committed_h >= args.budget_hours:
                    if not budget_exhausted:
                        _log_console(
                            f"[BUDGET-CAP] Committed GPU-hours ({committed_h:.2f}h) reached "
                            f"--budget-hours={args.budget_hours:.2f}h; draining in-flight jobs and stopping."
                        )
                        budget_exhausted = True
                    break

                deps = [str(d) for d in exp.get("depends_on", [])]
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

                exp_id = str(exp["id"])
                attempted_ids.add(exp_id)
                jobs_launched += 1
                fut = pool.submit(
                    _run_one_modal_experiment,
                    dict(exp),
                    queue_path=queue_path,
                    modal_exe=args.modal_exe,
                    runner_path=runner_path,
                    gpu=args.gpu,
                    timeout_secs=args.timeout_secs,
                    poll_secs=args.poll_secs,
                    hf_token=hf_token,
                    log_dir=args.log_dir,
                )
                running[fut] = exp
                with BUDGET_LOCK:
                    reserved_est_hours[fut] = est_h

            if not running:
                if budget_exhausted or (args.max_jobs > 0 and jobs_launched >= args.max_jobs):
                    break
                if waiting_deps == 0:
                    break
                _log_console(
                    f"[WAIT-DEPS] {waiting_deps} job(s) waiting on predecessors; re-checking in 30s..."
                )
                time.sleep(30)
                continue

            done, _ = fut_wait(running.keys(), timeout=30.0, return_when=FIRST_COMPLETED)
            for fut in done:
                exp = running.pop(fut)
                res = fut.result()
                with BUDGET_LOCK:
                    reserved_est_hours.pop(fut, None)
                    used_gpu_hours += float(res.get("elapsed_sec", 0.0)) / 3600.0
                if res.get("status") not in {"COMPLETED", "SKIPPED"}:
                    failures += 1

    _log_console(
        f"Modal {args.gpu} DAG worker finished. "
        f"jobs_launched={jobs_launched}, used_gpu_hours={used_gpu_hours:.2f}h, failures={failures}"
    )
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
