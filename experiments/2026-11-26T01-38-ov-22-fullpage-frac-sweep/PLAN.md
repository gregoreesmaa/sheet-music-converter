# Experiment Plan — `2026-11-26T01-38-ov-22-fullpage-frac-sweep`

> **Zero-Prior-Context Handoff:** Everything required to launch, monitor, resume, stop, or evaluate Overnight Experiment 22 is self-contained in this folder (`README.md`, `PLAN.md`, `run_colab.py`).

## 1. Platform Chosen & Rationale
- **Provider:** Google Colab (`google-colab-cli` via `.\.venv\Scripts\colab.exe`, Pro+ compute-unit pool).
- **Hardware:** `1x NVIDIA L4` (`24 GB` VRAM, Ada Lovelace `cc=8.9`, **native BF16**, `torch 2.14.1+cu130` pre-installed).
- **Session Name:** `ov-22-fp-frac`
- **Estimated Wall-Clock & Cost:** `~32-38 min` total (`~9-11 min` per `800`-step arm x 3 arms + `~4-5 min` `proxy_val` Raw + EMA evaluation) ≈ `~2.6-3.1 CU`.
- **Rationale:** Native BF16 + pre-installed PyTorch 2.14.1 gives fast ~8s cold starts with zero contention against the local RTX 4080 SUPER.

## 2. Identifiers for Follow-Up Queries
- **Colab Session:** `ov-22-fp-frac`
- **Hugging Face Repo (`dataset`):** `gregoreesmaa/sheet-music-converter-omr`
  - Eval JSON: `evals/2026-11-26T01-38-ov-22-fullpage-frac-sweep.json`
  - Run Ledger JSON: `runs/2026-11-26T01-38-ov-22-fullpage-frac-sweep.json`
- **Remote Paths on Colab VM:**
  - Code bundle: `/content/repo_bundle.tar.gz` -> extracted to `/workspace/sheet-music-converter`
  - Dataset & derived strip cache: `/content/vol/datasets/iter10_5_proxy_2k/` and `/content/vol/datasets/iter10_5_proxy_2k_strips/`
  - Rolling checkpoints & per-arm cached evals: `/content/vol/ckpts/2026-11-26T01-38-ov-22-fullpage-frac-sweep/<arm>/`
  - Transient token file: `/tmp/.hf_token` (`chmod 600`, deleted immediately after read AND guaranteed deleted in `finally`)

### Copy-Paste Monitoring & Retrieval Commands (PowerShell)
```powershell
# Check active Colab sessions & compute-unit balance
.\.venv\Scripts\colab.exe sessions
.\.venv\Scripts\colab.exe usage

# Stop Colab session immediately (ALWAYS run in finally!)
.\.venv\Scripts\colab.exe stop -s ov-22-fp-frac

# Fetch completed eval JSON from Hugging Face
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; p = hf_hub_download('gregoreesmaa/sheet-music-converter-omr', 'evals/2026-11-26T01-38-ov-22-fullpage-frac-sweep.json', repo_type='dataset'); print(open(p, encoding='utf-8').read())"
```

## 3. Required Training & Eval Data (100% Permissive, 0 Eval Overlap)
- **HF Shard:** `datasets/iter10_5_proxy_2k.tar` (`gregoreesmaa/sheet-music-converter-omr`)
  - `proxy_train.jsonl`: `2,000` deduplicated full-page OMR samples (stratified printed + synth HW + real HW; 0 `CC-BY-NC`, 0 `suite31_hw` overlap).
  - `proxy_val.jsonl`: `64` held-out full-page validation samples.
- **Derived System Strips (for `fp_070` and `fp_085`):** Deterministically sliced from the first `600` training pages of `proxy_train.jsonl` using uniform horizontal system bands (`N = len(bekerns)` with 8% vertical padding), written in `discover_strip_pairs` format (`<stem>_s{idx:02d}.png` + `<stem>_s{idx:02d}-bekern.txt`), split into `printed/` and `hw/` subdirectories. Zero val pages are ever touched.
- **Warm-Start:** None (identical scratch `seed=42` initialization across all 3 arms).

## 4. Exact Launch Command & Hyperparameters (Commit `164852f`)
```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\colab.exe new -s ov-22-fp-frac --gpu L4
.\.venv\Scripts\colab.exe upload -s ov-22-fp-frac <local_bundle.tar.gz> /content/repo_bundle.tar.gz
.\.venv\Scripts\colab.exe upload -s ov-22-fp-frac <local_tmp_token> /tmp/.hf_token
.\.venv\Scripts\colab.exe exec -s ov-22-fp-frac -f experiments/2026-11-26T01-38-ov-22-fullpage-frac-sweep/run_colab.py
.\.venv\Scripts\colab.exe stop -s ov-22-fp-frac
```

| Hyperparameter | Arm A (`fp_070`) | Arm B (`fp_085`) | Arm C (`fp_100`) |
|---|---|---|---|
| `--curriculum` | `mixed` | `mixed` | `fullpage` |
| `--fullpage-frac` *(isolated variable)* | **`0.70`** | **`0.85`** | **`1.00`** |
| `--preset` | `small_45m` | `small_45m` | `small_45m` |
| `--optimizer` | `muon` | `muon` | `muon` |
| `--steps` / `--seed` | `800` / `42` | `800` / `42` | `800` / `42` |
| `--batch-size` / `--grad-accum` / `--strip-batch-size` | `2` / `4` / `4` | `2` / `4` / `4` | `2` / `4` / `4` |
| `--fullpage-max-dim` / `--max-seq-len` | `1280` / `1536` | `1280` / `1536` | `1280` / `1536` |
| `--lr-schedule` / `--lr` / `--warmup-steps` | `wsd` / `3e-4` / `80` | `wsd` / `3e-4` / `80` | `wsd` / `3e-4` / `80` |
| `--augment --aug-prob` | `0.50` | `0.50` | `0.50` |
| `--grad-checkpoint --channels-last --bf16` | Enabled | Enabled | Enabled |
| `--save-every-minutes` | `5.0` | `5.0` | `5.0` |

## 5. 5-Minute Checkpointing & Resume Plan
- `train_uni_v2.py` runs with `--save-every-minutes 5.0`, atomically writing `uni_omr_v2_latest.pt` + `uni_omr_v2_ema_latest.pt` (`.tmp` -> `os.replace`) into `/content/vol/ckpts/2026-11-26T01-38-ov-22-fullpage-frac-sweep/<arm>/`.
- Each completed arm persists `/content/vol/ckpts/.../<arm>/arm_result.json`; re-executing `run_colab.py` in the same session automatically skips finished arms and resumes cleanly.

## 6. Evaluation Gate & Decision Tree
- **Metrics Measured (Raw & EMA checkpoints on `proxy_val.jsonl`, 64 full pages):**
  - Overall, `printed`-only, and `hw`-only teacher-forced **Val CE**, **Val MTP**, **Val Total Loss**.
  - KV-cached greedy **Val SER**, **Val Token F1**, and **Sys-Count Exact**.
  - Wall-clock training speed (`steps/sec`) and peak VRAM (`GB`).
- **Decision Tree:**
  - **PROMOTE:** Whichever `fullpage_frac` (`0.70`, `0.85`, or `1.00`) achieves the lowest EMA `proxy_val` SER and highest Token F1 (with >= `0.015` Token F1 margin or >= `0.02` lower Val CE over runners-up) becomes the locked Iter-11 `UniOMR v2` default `--fullpage-frac`.
  - **ITERATE:** If `0.85` and `1.00` tie within `< 0.01` Token F1, test a two-stage schedule (`0.75` for first 30% of steps -> `1.00` for final 70%) on a longer budget.
  - **DISCARD:** If `0.70` degrades both printed and HW full-page SER due to reduced full-page step exposure, drop `--fullpage-frac <= 0.70` from Iter-11 configs.
