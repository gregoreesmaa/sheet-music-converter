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
- **Passing `HF_TOKEN` into Kaggle kernels:** Kaggle CLI has no `secrets` API, so we maintain a **strictly private** Kaggle Dataset `gregoreesmaa/omr-hf-auth` containing `hf_token.txt`. Attach `"dataset_sources": ["gregoreesmaa/omr-hf-auth"]` in `"is_private": true` kernels, and read the token cleanly without ever logging it:
  ```python
  import glob, os
  tok = os.environ.get("HF_TOKEN", "").strip()
  if not tok:
      for p in glob.glob("/kaggle/input/**/hf_token.txt", recursive=True):
          tok = open(p).read().strip()
          if tok:
              break
  ```
  (Alternatively, if added in Web UI Settings → Secrets, `kaggle_secrets.UserSecretsClient().get_secret("HF_TOKEN")` also works.) **NEVER** print the token or set `"is_private": false` on any kernel attaching `omr-hf-auth`.
- **Check weekly quota anytime:** `.\.venv\Scripts\kaggle.exe quota`

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
- **2026-11-26 (Kernel slug/title mismatch causing `Permission 'kernels.get' was denied` on `ov-37` / `ov-39`):**
  - **Root cause:** `kaggle kernels push` derives the actual created kernel slug from `kernel-metadata.json`'s **`title`** field, **NOT** from `id`! When `id` was `gregoreesmaa/omr-ov-39-...` and `title` was `"Omr Omr Ov 39 ..."` (accidentally double-prefixing `OMR ` onto an `omr-` slug), Kaggle printed a warning (`Your kernel title does not resolve to the specified id`) and silently created `gregoreesmaa/omr-omr-ov-39-...` instead. Subsequent `kaggle kernels status gregoreesmaa/omr-ov-39-...` calls then failed with `Permission 'kernels.get' was denied` because that slug did not exist.
  - **Fix & invariant:** Always keep slugs short (e.g. `omr-ov-37`, `omr-ov-39`), set `"title": slug` (exact 1:1 match with `id` after `<username>/`), verify `rc == 0` and `"successfully pushed"` in `kernels push` stdout, parse the canonical `<user>/<slug>` from the pushed URL (`https://www.kaggle.com/code/<user>/<slug>`) before polling, and fail fast after consecutive `UNKNOWN` / permission errors instead of polling until timeout.
- **2026-11-26 (GPU slot limit returns `rc=0` on `kernels push`):**
  - When both concurrent free-tier GPU slots are already occupied, `kaggle kernels push` exits with **`rc=0`** and prints `Kernel push error: Maximum batch GPU session count of 2 reached.` to stdout (no non-zero exit code).
  - Always inspect stdout for `"Maximum batch GPU session count"` and treat it as a transient `[WAIT-SLOT]` (sleep 60s & retry) rather than a hard failure, and never rely solely on `rc == 0` to assume a push succeeded.
- **2026-11-26 (`ov-37` T4 silent FP32 fallback when gating on `cc[0] >= 8`):**
  - Scripts ported from Colab L4 that only write `use_bf16 = bool(cc[0] >= 8)` without an `elif` branch for `FP16 + torch.amp.GradScaler` silently run in **pure FP32** on Kaggle T4 (`cc=7.5`), ~2.2× slower (`78.5 min` vs `~35 min`) while leaving GPU 1 idle. Always provide an explicit `amp_dtype = torch.bfloat16 if cc[0] >= 8 else torch.float16` + `GradScaler(enabled=(amp_dtype == torch.float16))` path so T4 runs get Tensor Core speed without NaNs.
- **2026-11-27 (`i14-04` silent `missing=349` checkpoint load + 3.25 h wasted T4 burn):**
  - UniOMR v2 checkpoints store weights under `ck["model"]` and config under `ck["config"]`. Custom loaders that omit `"model"`/`"config"` and call `load_state_dict(..., strict=False)` silently evaluate a 100% random-init model (`missing=349, unexpected=7`), hitting `max_len` on every page and burning hours of Kaggle quota for `0.0000` F1.
  - **Invariant:** Always load checkpoints via canonical `eval_uni_suite31.load_uni_checkpoint(...)`, hard-assert `len(meta["missing_keys"]) == 0`, use canonical `uni_omr._pil_to_chw_tensor` (`[0, 1]`, never `[-1, 1]`), and add a 2-page fail-fast sanity gate (`dec_ok > 0` and `note_f1 >= 0.05`) at the start of Arm A.
