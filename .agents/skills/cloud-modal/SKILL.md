---
name: cloud-modal
description: >-
  Operational playbook for running sheet-music-converter OMR training, proxy
  ablations, and evals on Modal GPUs (L4/A10G/A100/H100, BF16 native) within
  the $30/mo free tier. Covers auth, CLI, GPU burn rates, image layer caching,
  Secrets, Volumes vs HF sync, detached runs, and empirical gotchas.
---

# Cloud Provider Playbook — Modal (`$30/mo` Free Tier)

**Trigger:** Activate alongside `test-in-cloud` whenever dispatching, debugging, or monitoring GPU workloads on Modal.

> [!IMPORTANT]
> **Living Skill Amendment Rule:** Whenever you hit a new Modal CLI flag, billing quirk, volume mount behavior, image build trap, or GPU issue, **immediately append it to `## Empirical Learnings & Gotchas`** at the bottom of this file before moving on.

---

## Auth & CLI Setup

- **Credentials file:** `~/.modal.toml` (auto-managed by `modal token set`). **NEVER** commit `~/.modal.toml` or paste Modal tokens into scripts/logs.
- **CLI binary (Windows `.venv`):** `.\.venv\Scripts\modal.exe`
- **Quick auth & profile check:**
  ```powershell
  .\.venv\Scripts\modal.exe profile current
  .\.venv\Scripts\modal.exe app list
  ```
- **Hugging Face Secret (one-time setup):**
  Store `HF_TOKEN` in a Modal Secret so containers pull/push private HF datasets without embedding tokens in code:
  ```powershell
  $hf = (Get-Content "$env:USERPROFILE\.cache\huggingface\token" -Raw).Trim()
  .\.venv\Scripts\modal.exe secret create hf-secret HF_TOKEN=$hf
  ```
  Attach in Python via `secrets=[modal.Secret.from_name("hf-secret")]`.

---

## GPU Tiers & Approximate Burn Rates (`$30/mo` Free Credit)

Modal bills per-second of active container runtime (zero idle cost). Approximate rates (verify current pricing in Modal dashboard if tight on credit):

| GPU | VRAM | BF16 Native | Approx. $/hr | Hours per `$30` | Best Use in `sheet-music-converter` |
|---|---:|---|---:|---:|---|
| `T4` | 16 GB | No (FP16 only) | ~$0.59 | ~50 h | Avoid on Modal (use free Kaggle `2× T4` instead) |
| `L4` | 24 GB | **Yes** | ~$0.80 | ~37 h | **Default workhorse** for `2k–4k` step proxy ablations & parallel sweeps |
| `A10G` | 24 GB | **Yes** | ~$1.10 | ~27 h | Fast `1280px–1536px` ablations needing higher memory bandwidth than L4 |
| `A100-40GB` | 40 GB | **Yes** | ~$2.10 | ~14 h | Large batch / `1536px` full-page `UniOMR v2` trials |
| `A100-80GB` | 80 GB | **Yes** | ~$2.50 | ~12 h | Memory-heavy context-length / multi-page experiments |
| `H100` | 80 GB | **Yes** (FP8 too) | ~$3.95 | ~7.5 h | High-leverage architecture sprints only — never waste on unverified code |

**Budget rule:** Always run a 2-minute `--max-steps 50` smoke test on `L4` before fanning out 4–5 parallel trials or booking an `A100`/`H100`.

---

## Image Caching Best Practices (2-Second Code Rebuilds)

Modal caches `modal.Image` layers top-down. Keep slow, static installs at the top and frequently changing repo files at the bottom:

```python
import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "tar")
    .pip_install(
        "torch==2.5.1",
        "torchvision==0.20.1",
        "huggingface_hub>=0.26",
        "pillow",
        "numpy",
        "tqdm",
    )
    # Put local repo code LAST so edits rebuild in ~2s instead of re-installing PyTorch
    .add_local_dir("tmp-train/mvd", remote_path="/workspace/mvd")
)
```

- Pin `torch` + CUDA wheel versions explicitly so Modal cache keys stay stable across runs.
- Keep `datasets/` and checkpoints **out** of `modal.Image` — store them on a Modal Volume or pull from Hugging Face.

---

## Modal Volume vs. Hugging Face Sync

Use **both** cleanly — they solve different problems:

| Layer | Role | Lifecycle |
|---|---|---|
| **Modal Volume** (`modal.Volume.from_name("omr-cache", create_if_missing=True)`) | Fast local SSD cache for unpacked dataset shards (`/vol/datasets/`) and 5-minute rolling checkpoints (`/vol/ckpts/<run_id>/`). Avoids re-downloading 10 GB of shards on every parallel trial. | Persist across Modal runs; call `vol.commit()` after writing checkpoints so a killed container doesn't lose them. |
| **Hugging Face Hub** (`cloud-huggingface`) | Cross-provider source of truth (`datasets/`, `manifests/`, `checkpoints/`, `evals/`, `runs/`). | Sync rolling checkpoints to HF every 15–30 min and on run exit; upload final `suite31_hw` eval JSON immediately. |

---

## Detached Execution, Timeouts & Monitoring

- Always set an explicit `timeout` on `@app.function(gpu="L4", timeout=3600, ...)` (default is 300s, which will kill training!). Max Modal function timeout is 24 hours (`86400`).
- Launch long runs **detached** so closing your local terminal or losing Wi-Fi never kills the GPU job:
  ```powershell
  .\.venv\Scripts\modal.exe run --detach scripts/cloud/modal_train.py --run-id v2-ablation-01 --max-steps 4000
  ```
- Inspect running apps, tail logs, or cancel runaway jobs:
  ```powershell
  .\.venv\Scripts\modal.exe app list
  .\.venv\Scripts\modal.exe app logs <app-id>
  .\.venv\Scripts\modal.exe app stop <app-id>
  ```
- Remember our PyTorch SDPA rule on Linux/Modal too: avoid forcing `CUDNN_ATTENTION` on variable `(mem, T)` batches; prefer `EFFICIENT_ATTENTION` / FlashAttention where supported.

---

## Empirical Learnings & Gotchas

> Append verified observations here with date (`YYYY-MM-DD`) whenever encountered.

- **2026-11-26 (Live 1-min GPU smoke test, profile `gregoreesmaa1`):**
  - First-time `debian_slim` Python 3.12 + `torch-2.14.1+cu126` image build took **`65.2s`** (400+ MB/s wheel download, cached thereafter); container cold-start + torch import took **`~11.6s`**; T4 (`14.57 GiB`, `cc=7.5`) ran `16.04 TFLOPS` FP16.
  - **NumPy warning in slim images:** PyTorch emits `UserWarning: Failed to initialize NumPy` unless `numpy` is explicitly listed in `.pip_install(...)` — always include `numpy` in training images.
  - **CRITICAL BF16 emulation trap on T4:** `torch.cuda.is_bf16_supported()` returns **`True`** on Turing T4 (`cc=7.5`) in PyTorch 2.14.1+cu126 via slow emulation! Always gate native BF16 on `torch.cuda.get_device_capability()[0] >= 8` (or `torch.cuda.is_bf16_supported(including_emulation=False)`), and prefer **L4 (`cc=8.9`) / A10G (`cc=8.6`)** on Modal for BF16 runs.
  - **Billing safety:** Free Starter tier (`$30/mo`) hard-stops if no card is attached; if a card is on file, set **Modal Dashboard → Settings → Billing → Spending Limit = `$30`** (`$0` overage) to guarantee hard cutoff.
