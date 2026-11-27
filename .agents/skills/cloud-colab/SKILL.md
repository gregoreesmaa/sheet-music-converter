---
name: cloud-colab
description: >-
  Operational playbook for running sheet-music-converter OMR training, proxy
  ablations, and OOD evals on Google Colab GPUs (T4/L4/A100/H100) via the
  official `google-colab-cli` (`.\.venv\Scripts\colab.exe`). Covers OAuth2 auth,
  compute unit budgets, session lifecycle, HF Private Hub sync, and empirical
  Windows/Colab gotchas.
---

# Cloud Provider Playbook — Google Colab (`google-colab-cli`)

**Trigger:** Activate alongside `test-in-cloud` whenever dispatching, debugging, or monitoring GPU workloads on Google Colab.

> [!IMPORTANT]
> **Living Skill Amendment Rule:** Whenever you hit a new Colab CLI quirk, compute-unit burn rate, GPU availability issue, or session/auth trap, **immediately append it to `## Empirical Learnings & Gotchas`** at the bottom of this file before moving on.

---

## Auth & CLI Setup

- **CLI binary (Windows `.venv`):** `.\.venv\Scripts\colab.exe` (`google-colab-cli==0.8.0`, installed via `uv pip install --python .\.venv\Scripts\python.exe google-colab-cli`).
- **Auth strategy:** Default `--auth oauth2` (cached locally in `~/.config/colab-cli/`). **NEVER** commit or print OAuth tokens or `HF_TOKEN`.
- **Quick balance & session check:**
  ```powershell
  .\.venv\Scripts\colab.exe usage
  .\.venv\Scripts\colab.exe ls
  ```
  *(Note: the list subcommand is `colab ls`, NOT `colab list` or `colab sessions`.)*

---

## Compute Unit Budget & GPU Tiers

Verified balance on `2026-11-26`: **`1,479.59` compute units** (Pro/Pro+ pool).

| GPU | VRAM | BF16 Native | Approx. CU/hr | Best Use in `sheet-music-converter` |
|---|---:|---|---:|---|
| `T4` | 16 GB | **No** (`cc=7.5`, FP16-only) | ~1.9 | Cheap CPU/GPU data prep or FP16 inference only (avoid for BF16 training) |
| `L4` | 24 GB | **Yes** (`cc=8.9`) | **2.33** *(measured)* | Fast proxy ablations, `suite31_hw` OOD evals, overnight DAG queue (`~635` L4-hrs on 1,478 CU) |
| `A100` | 40 GB | **Yes** (`cc=8.0`) | ~11–13 | Full `1536px` `UniOMR v2` training runs (Iter 11+) & large-batch RLVR |
| `H100` | 80 GB | **Yes** (`cc=9.0`) | ~15–18 | High-leverage architecture sprints once code is 100% verified on L4 |

**Budget rule:** Always run a 10–60s smoke test on `L4` before booking `A100`/`H100`, and always pass `--save-every-minutes 5.0` with HF Private Hub checkpoint syncing so a disconnected runtime never loses more than 5 minutes of work.

---

## Running Jobs & Syncing with Hugging Face Private Hub

1. **Passing `HF_TOKEN` securely (zero secrets in scripts or git):**
   Never hardcode `HF_TOKEN` inside a script. Pass it at runtime via a temporary untracked env file or stdin/upload into `/tmp/.hf_token` on the Colab VM (`chmod 600`), read it inside Python, and delete `/tmp/.hf_token` on exit.
2. **Ephemeral one-shot script (`colab run`):**
   Provisions a fresh VM, runs the script, and automatically tears down the VM on exit (zero idle CU burn!):
   ```powershell
   .\.venv\Scripts\colab.exe run --gpu L4 experiments/<exp_id>/run_colab.py
   ```
3. **Named persistent session (`colab new` / `exec` / `stop`):**
   Useful for multi-step workflows (upload repo tarball → install deps → run training → download logs → `colab stop -s <name>`):
   ```powershell
   .\.venv\Scripts\colab.exe new -s iter11-a100 --gpu A100
   .\.venv\Scripts\colab.exe upload -s iter11-a100 local_bundle.tar.gz /content/bundle.tar.gz
   .\.venv\Scripts\colab.exe exec -s iter11-a100 -f experiments/<exp_id>/run_colab.py
   .\.venv\Scripts\colab.exe stop -s iter11-a100
   ```
   **ALWAYS `colab stop -s <name>`** when finished so you never burn idle compute units!

---

## Empirical Learnings & Gotchas

> Append verified observations here with date (`YYYY-MM-DD`) whenever encountered.

- **2026-11-26 (Live Windows `L4` smoke test via `colab.exe 0.8.0`):**
  - **Works natively on Windows** despite the README claiming Linux/macOS-only: `.\.venv\Scripts\colab.exe run --gpu L4` provisioned an `NVIDIA L4` (`23,034 MiB`, `cc=(8, 9)`, `bf16_native=True`) with **`torch 2.14.1+cu130` pre-installed** in **`~8 seconds`** and tore down cleanly with exit code `0`.
  - **UTF-8 BOM trap on Windows PowerShell 5.1:** `Set-Content -Encoding utf8` writes a leading `U+FEFF` BOM, which causes `SyntaxError: invalid non-printable character U+FEFF` inside Colab's IPython kernel when passed to `colab run` / `colab exec -f`. Always write Python scripts as **UTF-8 without BOM** (`write_file` tool, Python `open(..., encoding="utf-8")`, or `[System.IO.File]::WriteAllText(...)`).
  - **No `pip install torch` needed:** Colab VMs ship with `torch 2.14.1+cu130` pre-installed (matching our local RTX 4080 SUPER version!), saving ~60s per cold start. Only lightweight packages (`huggingface_hub`, `verovio`, etc.) need `colab install` / `uv pip install`.
  - **`colab exec` 30s idle-output timeout trap:** `colab exec` defaults to `--timeout 30` seconds of silence between iopub stdout messages and throws `TimeoutError` if training logs less often than every 30s. Always pass `--timeout 600` (or higher) AND print flushed progress every 25–50 steps.
  - **Concurrent GPU pool caps:** Colab Pro+ caps concurrent `L4` sessions at 2; when 2 `L4` sessions are occupied, `A100` can still be provisioned as a separate pool (`NVIDIA A100-SXM4-40GB`, `1.80 it/s` on `small_45m` `1280px` BF16). Always check `colab sessions` and tear down in `finally`.
- **2026-11-26 (Colab CLI `--timeout` & per-tier concurrency cap):**
  - **`colab exec` 30s silence timeout:** `colab exec` defaults to `--timeout 30` (30 seconds without stdout/iopub messages raises `TimeoutError`). Always pass `--timeout 1800` (or `>= 600`) to `colab exec` for training/eval scripts AND flush progress every ~20–25 steps.
  - **Per-GPU-tier concurrency cap:** Colab Pro+ allows up to **2 simultaneous `L4` sessions** (a 3rd `L4` returns `Allocation refused (precondition failed)`), while `A100` remains independently allocatable when L4 slots are full.
  - **2-session concurrent `L4` cap (`Allocation refused`):** Colab caps concurrent `L4` assignments at **2**; requesting a 3rd `L4` immediately returns `exit_code=1` with `Allocation refused (precondition failed)`. Always cap parallel `L4` workers at `--max-parallel 2` and retry `colab new` on a 60s backoff when `Allocation refused` is seen instead of marking a queued job `FAILED`.
  - **2026-11-26 (`colab exec` idle timeout trap on Exp 6):** `colab exec` defaults to `--timeout 30.0` (seconds without stdout before failing). Always pass `--timeout 0` (no limit) for builds, downloads, or long GPU loops, and wrap `colab exec` in a `try { ... } finally { .\.venv\Scripts\colab.exe stop -s <name> }` block so the VM never lingers idle after completion or error.
- **2026-11-26 (Colab concurrency & `exec --timeout` gotchas):**
  - **Per-tier GPU concurrency cap:** When 2 `L4` sessions are already active, `colab new --gpu L4` returns `Allocation refused (precondition failed)`. Falling back to `--gpu A100` provisions cleanly on A100-SXM4-40GB (separate tier, `cc=(8, 0)`, BF16 native, Rust release builds of `assembly-serialize` + `eval-harness` finish in `31.8s`).
  - **`colab exec` default 30s silence timeout:** `colab exec` raises `TimeoutError: no message from the kernel for 30s` if a cell is quiet for >30s. ALWAYS pass `--timeout 600` (or higher) to `colab exec` for training/eval scripts!
  - **`importlib.invalidate_caches()` after in-cell `pip install`:** If a script checks `__import__(pkg)` (triggering `ImportError`), then runs `pip install -q <pkg>`, you MUST call `importlib.invalidate_caches()` before importing modules that optionally import `<pkg>`, or Python's finder cache silently returns `ImportError` (`render_ok_rate = 0.0` trap!). Always add a fail-fast `assert` smoke test after runtime installs!
  - **`verovio.toolkit()` inside `ThreadPoolExecutor` worker threads:** Calling `verovio.toolkit()` inside worker threads on Linux Colab silently fails to locate `verovio/data` fonts (`render_ok_rate = 0.0` even when main-thread smoke render passes!). Always use a single shared `verovio.toolkit(False)` with explicit `tk.setResourcePath(str(Path(verovio.__file__).resolve().parent / "data"))` guarded by a `threading.Lock()`.
  - **`verovio.toolkit()` C++ font-table race under `ThreadPoolExecutor` (2026-11-26, Academic Exp A):** Instantiating `verovio.toolkit()` concurrently in worker threads triggers `[Error] Bravura font could not be loaded` / `[Error] Leipzig font could not be loaded` and returns `None` for 100% of threads (`render_ok_rate = 0.0`). Always serialize `verovio.toolkit()` calls through a single module-level instance guarded by `threading.Lock()` (fixed in `tmp-train/mvd/dual_verifier.py`), leaving `resvg_py` SVG→PNG rasterization unlocked in parallel.
- **2026-11-26 (Exp 8 `L4` benchmark via `colab.exe 0.8.0`):**
  - **`colab exec` has NO `-c` flag:** Pass inline code as a positional argument (`colab exec -s <name> "print(1)"`) or pass a script file with `-f <path>`.
  - **`colab install` prefers `uv` automatically:** Installing `onnx==1.23.2 onnxruntime==1.31.1 onnx-ir onnxscript huggingface_hub` took only `~13 s` on an `L4` VM.
  - **Full Exp 8 (`small_45m` + `edge85m` FP32/INT4 parity across 12 `suite31_hw` pages + 6-file multi-target ONNX export) completed in `72.9 s`** (`~0.10 CU`) and `colab stop -s <name>` terminated cleanly in `~2 s`.
  - **Unbounded measure-beam sweep runtime (`ov-34`):** Sweeping 9 unbudgeted `measure_beam_size ∈ {2,4,8}` configs across 90 `suite31_hw` pages at `1536px/4096` took **`8640 s` (`144 min`, `~11.5 CU`)** vs the `20 min` estimate because noisy HW pages triggered 10–72 bar rewinds each (`+102%` to `+280%` HW decode latency). Always enforce a per-page trigger cap (`max_triggers_per_page <= 6`) and budget `~6–8 min/config` for full 90-page `suite31_hw` decode sweeps on `L4`.
  - **Passive multi-shadow EMA + post-hoc souping saves 3× GPU time (`ov-15`):** Because `GPUEmaShadow` is a passive shadow copy that never alters training gradients, sweeping 3 EMA decays in 3 separate 800-step runs burned `5,739 s` (`95.6 min`) of redundant training (`8,461 s` = `141.0 min` total session, `~2.39 s/step` + `~4.5 min/eval`). Attach multiple passive shadow trackers (`~172 MB` VRAM each for `small_45m`) and snapshot hooks to a **single** training trajectory to eliminate both redundant compute and run-to-run CUDA greedy-decode variance.
- **2026-11-26 (silent false-COMPLETED jobs + queue clobbering, found via ov-16..21):**
  - **`colab exec` returns exit 0 even when the in-VM job crashed** (`SystemExit` is swallowed by the IPython kernel). Never trust the outer exit code: the bootstrap prints `[bootstrap:<id>] Finished rc=N` and `run_queue_colab.py` now parses that marker (missing marker -> FAILED rc=97). Five jobs (ov-16/17/19/20/21) had been recorded COMPLETED with no HF eval.
  - **`subprocess.run` with inherited fds inside `colab exec` loses ALL child stdout/stderr** (no tracebacks, no progress). The bootstrap now uses `Popen(stdout=PIPE, stderr=STDOUT)` and re-prints each line. Always verify COMPLETED against `evals/<id>.json` on HF.
  - **A long-running queue worker must re-read `queue.json` under `.queue.lock` before every write.** An old worker holding a startup snapshot overwrote the file and silently dropped 12 analyst-appended ITERATE follow-ups. After upgrading worker code, restart the worker (a running process keeps its old code). `experiments/overnight_queue/restore_followups.py` / `fix_queue_state.py` rebuild lost entries from the experiment folders.
  - **Killing the worker does not stop its Colab sessions or in-kernel jobs**: `ov-22`/`ov-24` kept computing after the worker died. Find orphans with `colab ls` and stop them with `colab stop -s <name>` once their HF eval lands.
  - **Linux `eval_uni_suite31.py` silent `0/45` skip (`assembly-serialize.exe`)**: Prior to 2026-11-26, `tmp-train/mvd/eval_sys_gate.py` hardcoded `.exe` on `ASER`/`EHARNESS`, causing `ov-33` and `i13-04` to skip 45/45 pieces on Linux with `rc=0` and all-`null` summaries. Fixed via `_BIN_EXT = '.exe' if os.name == 'nt' else ''`, and follow-up runners (`ov-33b`) now assert `all45_clean.note_f1 is not None` fail-closed.
  - **Transient `colab_cli` websocket drops (`RuntimeError: Connection was lost.` in `colab_cli/runtime.py:305`)**: Seen on `i13-25` at step `250/400` after 10 min of clean L4 training (no Python/CUDA error), and on `i13-16b` (`2026-11-26T21:32Z`) at cell startup. **CRITICAL Windows hang gotcha:** after printing `RuntimeError: Connection was lost.`, `colab.exe exec` can leave a zombie local Python process hanging indefinitely on Windows without exiting — blocking a `run_queue_colab.py` worker slot AND leaving the remote L4 VM running idle at `2.33 CU/hr`. When a session log ends with `RuntimeError: Connection was lost.`, always run `colab stop -s <name>` AND kill the hung local `colab.exe exec -s <name>` PID so the queue worker unblocks immediately.
  - **Silent multi-hour `suite31_hw` evals inside `run_colab.py` (`i14-05`):** Running 3×90-page `model.generate(max_len=2048)` passes per arm (`RAW`, `EMA`, `INT4-dequant` = `270` pages/arm) on an `L4` takes `~2 hours/arm` (`~28 s/page` for `small_45m`) with zero stdout if no per-page progress is printed. Do NOT assume a session is hung just because stdout pauses after `step=1200/1200` — check whether a quick `colab exec --timeout 20` queues behind a busy kernel before touching it, and always print flushed progress every 10–15 pages in future eval loops.
