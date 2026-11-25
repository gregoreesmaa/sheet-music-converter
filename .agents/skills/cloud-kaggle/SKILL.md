---
name: cloud-kaggle
description: >-
  Operational playbook for running sheet-music-converter OMR training,
  background ablations, and data prep on Kaggle's free 30 hrs/wk 2x T4 GPU
  notebooks/scripts. Covers auth, CLI, kernel-metadata.json, CRITICAL Turing T4
  FP16-only gotcha, 12h runtime limit, 20 GB disk budget, and empirical gotchas.
---

# Cloud Provider Playbook — Kaggle (`30 hrs/wk` Free, `2× T4` 32 GB)

**Trigger:** Activate alongside `test-in-cloud` whenever pushing, polling, or retrieving outputs from Kaggle GPU kernels.

> [!IMPORTANT]
> **Living Skill Amendment Rule:** Whenever you hit a new Kaggle CLI quirk, API auth change, quota limit, disk-full failure, or T4 precision bug, **immediately append it to `## Empirical Learnings & Gotchas`** at the bottom of this file before moving on.

---

## Auth & CLI Setup

- **Credentials:** `KAGGLE_API_TOKEN` stored in `~/.secrets/kaggle-token.txt` or `~/.kaggle/access_token` (or legacy `~/.kaggle/kaggle.json`). **NEVER** commit tokens or bake them into `kernel-metadata.json` or scripts.
- **CLI binary (Windows `.venv`):** `.\.venv\Scripts\kaggle.exe`
- **Load token in PowerShell before invoking CLI:**
  ```powershell
  $env:KAGGLE_API_TOKEN = (Get-Content "$env:USERPROFILE\.secrets\kaggle-token.txt" -Raw).Trim()
  .\.venv\Scripts\kaggle.exe kernels list --mine
  ```
- **Core CLI workflow:**
  ```powershell
  .\.venv\Scripts\kaggle.exe kernels push -p <kernel_dir>
  .\.venv\Scripts\kaggle.exe kernels status <username>/<kernel-slug>
  .\.venv\Scripts\kaggle.exe kernels output <username>/<kernel-slug> -p <out_dir>
  ```
- **Passing `HF_TOKEN` into Kaggle kernels:** Store `HF_TOKEN` in **Kaggle User Secrets** (Settings → Secrets → `HF_TOKEN`) and read it inside the script via `kaggle_secrets.UserSecretsClient().get_secret("HF_TOKEN")` (with `os.environ.get("HF_TOKEN")` fallback). Never hardcode it in the pushed `.py` file.

---

## `kernel-metadata.json` Template

Place this alongside your entrypoint script (e.g. `train_kaggle.py`) inside `<kernel_dir>`:

```json
{
  "id": "<kaggle-username>/<kernel-slug>",
  "title": "<Human Readable Kernel Title>",
  "code_file": "train_kaggle.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": true,
  "dataset_sources": [],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
```

- Always set `"is_private": true` (our training logs/manifests stay private).
- Always set `"enable_internet": true` so the kernel can pull dataset shards & push checkpoints/evals via Hugging Face Hub.
- To select `GPU T4 x2` vs `GPU P100`, verify the accelerator setting on the kernel (or via CLI/UI) and log the exact `machine_shape` behavior in `Empirical Learnings & Gotchas`.

---

## CRITICAL: Turing T4 Precision Gotcha (`FP16` ONLY — NEVER `--bf16`!)

> [!CAUTION]
> **NVIDIA T4 (Turing, SM 7.5) does NOT have native BF16 tensor cores.**
> - Passing `--bf16` on Kaggle T4 either **crashes** (`RuntimeError` / unsupported dtype in Flash/CUTLASS kernels) or **silently falls back to slow FP32 emulation**, destroying throughput.
> - **ALWAYS use `--fp16` (with `torch.amp.GradScaler`) or `--fp32` on Kaggle T4.**
> - Keep `EFFICIENT_ATTENTION` / math SDPA fallback tested under FP16 (check for `NaN` loss spikes on early steps and enable gradient clipping `--grad-clip 1.0`).
> - When using **`2× T4` (16 GB + 16 GB = 32 GB)**, launch with `torchrun --nproc_per_node=2` (DDP) rather than single-GPU if batch size or throughput benefits — remember per-GPU VRAM is still `16 GB`.

---

## Runtime, Quota & Disk Limits

| Constraint | Limit | Required Mitigation |
|---|---|---|
| **Session timeout** | **12 hours** hard kill | Pass a soft budget (e.g. `--max-minutes 660`, 11 hours) so the script cleanly saves `latest.pt`, syncs to HF, and updates `runs/<run_id>.json` (`status: "preempted"` or `"trained"`) with 60 min margin. |
| **Weekly GPU quota** | **30 hours / week** (rolling) | Note: `2× T4` counts against the same 30 hr/wk quota (verify whether 1× or 2× wall-clock rate in `Empirical Learnings`). Don't burn quota on unverified scripts — smoke-test 5 min first. |
| **Working disk (`/kaggle/working`)** | **20 GB** max | Stream/download only the shards needed for the run into `/tmp` (larger ephemeral scratch) or unpack and **immediately delete `.tar` archives** after extraction. Keep only `latest.pt` + `best.pt` locally; upload to HF and prune older checkpoints. |
| **Output artifact size** | Limited on `kernels output` | Do NOT rely on Kaggle kernel outputs for large checkpoints — always push checkpoints and `suite31_hw` eval JSONs directly to **Hugging Face Hub** (`cloud-huggingface`). |

---

## Empirical Learnings & Gotchas

> Append verified observations here with date (`YYYY-MM-DD`) whenever encountered.

- **2026-11-26 (Live 1-min GPU smoke test, username `gregoreesmaa`, `omr-gpu-smoke-test`):**
  - **Auth:** Modern `KGAT_...` token saved in `~/.kaggle/access_token` works directly with **no** `KAGGLE_API_TOKEN` env var and **no** legacy `kaggle.json`. Discover username via `kaggle config view` (`auth_method: ACCESS_TOKEN`) — `/api/v1/users/me` returns HTTP 404.
  - **Hardware & Stack:** `enable_gpu: true` automatically provisions **2× Tesla T4** (`14.56 GiB` each, `cc=7.5`, `21.17 TFLOPS` FP16 on GPU 0), Driver `580.178.04`, CUDA `13.0`, Python `3.13.16`, PyTorch `2.11.0+cu130`. End-to-end queue + run completed in **`~25s`**.
  - **CRITICAL BF16 emulation trap on Kaggle T4:** `torch.cuda.is_bf16_supported()` returns **`True`** on Kaggle's T4 (`cc=7.5`) under PyTorch `2.11.0+cu130` via slow software emulation! ALWAYS check `torch.cuda.get_device_capability()[0] >= 8` (or `including_emulation=False`) and force **FP16 + `GradScaler`** on T4!
  - **Output log format:** `kaggle kernels output` downloads `<slug>.log` as a **JSON array** of `{stream_name, time, data}` objects (not plain text).
  - **Billing safety:** Kaggle free tier (`30 hrs/wk`) has **no card on file** and enforces a hard `$0` cutoff (GPU sessions simply refuse to start once weekly quota is reached). Never link a GCP billing account under *Add-ons → Google Cloud Services*.
