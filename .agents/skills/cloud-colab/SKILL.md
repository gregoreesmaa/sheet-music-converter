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
  .\.venv\Scripts\colab.exe sessions
  ```

---

## Compute Unit Budget & GPU Tiers

Verified balance on `2026-11-26`: **`1,479.59` compute units** (Pro/Pro+ pool).

| GPU | VRAM | BF16 Native | Approx. CU/hr | Best Use in `sheet-music-converter` |
|---|---:|---|---:|---|
| `T4` | 16 GB | **No** (`cc=7.5`, FP16-only) | ~1.9 | Cheap CPU/GPU data prep or FP16 inference only (avoid for BF16 training) |
| `L4` | 24 GB | **Yes** (`cc=8.9`) | ~4.8 | Fast proxy ablations, `suite31_hw` OOD evals, Wave 3 experiments |
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
- **2026-11-26 (Colab CLI `--timeout` & per-tier concurrency cap):**
  - **`colab exec` 30s silence timeout:** `colab exec` defaults to `--timeout 30` (30 seconds without stdout/iopub messages raises `TimeoutError`). Always pass `--timeout 1800` (or `>= 600`) to `colab exec` for training/eval scripts AND flush progress every ~20–25 steps.
  - **Per-GPU-tier concurrency cap:** Colab Pro+ allows up to **2 simultaneous `L4` sessions** (a 3rd `L4` returns `Allocation refused (precondition failed)`), while `A100` remains independently allocatable when L4 slots are full.
  - **2-session concurrent `L4` cap (`Allocation refused`):** Colab caps concurrent `L4` assignments at **2**; requesting a 3rd `L4` immediately returns `exit_code=1` with `Allocation refused (precondition failed)`. Always cap parallel `L4` workers at `--max-parallel 2` and retry `colab new` on a 60s backoff when `Allocation refused` is seen instead of marking a queued job `FAILED`.
  - **2026-11-26 (`colab exec` idle timeout trap on Exp 6):** `colab exec` defaults to `--timeout 30.0` (seconds without stdout before failing). Always pass `--timeout 0` (no limit) for builds, downloads, or long GPU loops, and wrap `colab exec` in a `try { ... } finally { .\.venv\Scripts\colab.exe stop -s <name> }` block so the VM never lingers idle after completion or error.
- **2026-11-26 (Colab concurrency & `exec --timeout` gotchas):**
  - **Per-tier GPU concurrency cap:** When 2 `L4` sessions are already active, `colab new --gpu L4` returns `Allocation refused (precondition failed)`. Falling back to `--gpu A100` provisions cleanly on A100-SXM4-40GB (separate tier, `cc=(8, 0)`, BF16 native, Rust release builds of `assembly-serialize` + `eval-harness` finish in `31.8s`).
  - **`colab exec` default 30s silence timeout:** `colab exec` raises `TimeoutError: no message from the kernel for 30s` if a cell is quiet for >30s. ALWAYS pass `--timeout 600` (or higher) to `colab exec` for training/eval scripts!
- **2026-11-26 (Exp 8 `L4` benchmark via `colab.exe 0.8.0`):**
  - **`colab exec` has NO `-c` flag:** Pass inline code as a positional argument (`colab exec -s <name> "print(1)"`) or pass a script file with `-f <path>`.
  - **`colab install` prefers `uv` automatically:** Installing `onnx==1.23.2 onnxruntime==1.31.1 onnx-ir onnxscript huggingface_hub` took only `~13 s` on an `L4` VM.
  - **Full Exp 8 (`small_45m` + `edge85m` FP32/INT4 parity across 12 `suite31_hw` pages + 6-file multi-target ONNX export) completed in `72.9 s`** (`~0.10 CU`) and `colab stop -s <name>` terminated cleanly in `~2 s`.
