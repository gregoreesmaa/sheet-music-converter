# Experiment Plan — `2026-11-26T01-38-ov-24-mixstyle-domain-invariance`

> **Zero-Prior-Context Handoff:** Everything required to launch, monitor, resume, stop, or evaluate Overnight Experiment 24 is self-contained in this folder (`README.md`, `PLAN.md`, `run_colab.py`).

## 1. Platform Chosen & Rationale
- **Provider:** Google Colab (`google-colab-cli` via `.\.venv\Scripts\colab.exe`, Pro+ compute-unit pool).
- **Hardware:** `1x NVIDIA L4` (`24 GB` VRAM, Ada Lovelace `cc=8.9`, **native BF16**, `torch 2.14.1+cu130` pre-installed).
- **Session Name:** `ov-24-mixstyle`
- **Estimated Wall-Clock & Cost:** `~33-39 min` total (`~9-10 min` per `800`-step arm x 3 arms + `~5-7 min` for `proxy_val` + 90-page `suite31_hw` zero-shot cross-domain eval) ≈ `~2.7-3.2 CU`.
- **Rationale:** Native BF16 + 24 GB VRAM runs all 3 arms cleanly in a single L4 session with zero local RTX 4080 SUPER contention.

## 2. Identifiers for Follow-Up Queries
- **Colab Session:** `ov-24-mixstyle`
- **Hugging Face Repo (`dataset`):** `gregoreesmaa/sheet-music-converter-omr`
  - Eval JSON: `evals/2026-11-26T01-38-ov-24-mixstyle-domain-invariance.json`
  - Run Ledger JSON: `runs/2026-11-26T01-38-ov-24-mixstyle-domain-invariance.json`
- **Remote Paths on Colab VM:**
  - Code bundle: `/content/repo_bundle.tar.gz` -> extracted to `/workspace/sheet-music-converter`
  - Datasets: `/content/vol/datasets/{iter10_5_proxy_2k,suite31_hw_eval}/`
  - Rolling checkpoints & per-arm evals: `/content/vol/ckpts/2026-11-26T01-38-ov-24-mixstyle-domain-invariance/<arm>/`
  - Transient token file: `/tmp/.hf_token` (`chmod 600`, deleted immediately after read AND guaranteed deleted in `finally`)

### Copy-Paste Monitoring & Retrieval Commands (PowerShell)
```powershell
# Check active Colab sessions & compute-unit balance
.\.venv\Scripts\colab.exe sessions
.\.venv\Scripts\colab.exe usage

# Stop Colab session immediately (ALWAYS run in finally!)
.\.venv\Scripts\colab.exe stop -s ov-24-mixstyle

# Fetch completed eval JSON from Hugging Face
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; p = hf_hub_download('gregoreesmaa/sheet-music-converter-omr', 'evals/2026-11-26T01-38-ov-24-mixstyle-domain-invariance.json', repo_type='dataset'); print(open(p, encoding='utf-8').read())"
```

## 3. Required Training & Eval Data (100% Permissive, 0 Eval Overlap)
- **Training Split (Printed-Only for Strict Zero-Shot Printed->HW Transfer):** Filtered printed-only rows from `datasets/iter10_5_proxy_2k.tar` (`proxy_train_printed_only.jsonl`), ensuring **zero** handwritten pages are seen during training so any HW gain is purely attributable to `MixStyle2d` domain invariance.
- **Validation & OOD Cross-Domain Splits:**
  - `proxy_val.jsonl` (64 pages): split into `printed` (in-domain) vs `hw` (zero-shot cross-domain).
  - `datasets/suite31_hw_eval.tar` (45 pieces x 2 = 90 pages): `printed_clean`, `printed_photo`, `hw_clean`, `hw_photo`.
- **Warm-Start:** None (identical scratch `seed=42` initialization across all 3 arms; `MixStyle2d` is 0-parameter so architectures are weight-identical).

## 4. Exact Launch Command & Hyperparameters (Commit `164852f`)
```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\colab.exe new -s ov-24-mixstyle --gpu L4
.\.venv\Scripts\colab.exe upload -s ov-24-mixstyle <local_bundle.tar.gz> /content/repo_bundle.tar.gz
.\.venv\Scripts\colab.exe upload -s ov-24-mixstyle <local_tmp_token> /tmp/.hf_token
.\.venv\Scripts\colab.exe exec -s ov-24-mixstyle -f experiments/2026-11-26T01-38-ov-24-mixstyle-domain-invariance/run_colab.py
.\.venv\Scripts\colab.exe stop -s ov-24-mixstyle
```

| Hyperparameter | Arm A (`ms_00`) | Arm B (`ms_20`) | Arm C (`ms_40`) |
|---|---|---|---|
| `--mixstyle-prob` *(isolated variable)* | **`0.0`** | **`0.20`** | **`0.40`** |
| `--preset` / `--optimizer` | `small_45m` / `muon` | `small_45m` / `muon` | `small_45m` / `muon` |
| Training manifest | `proxy_train_printed_only.jsonl` | Identical | Identical |
| `--steps` / `--seed` | `800` / `42` | `800` / `42` | `800` / `42` |
| `--batch-size` / `--grad-accum` | `2` / `4` | `2` / `4` | `2` / `4` |
| `--fullpage-max-dim` / `--max-seq-len` | `1280` / `1536` | `1280` / `1536` | `1280` / `1536` |
| `--lr-schedule` / `--lr` / `--warmup-steps` | `wsd` / `3e-4` / `80` | `wsd` / `3e-4` / `80` | `wsd` / `3e-4` / `80` |
| `--augment --aug-prob` | `0.50` | `0.50` | `0.50` |
| `--grad-checkpoint --channels-last --bf16` | Enabled | Enabled | Enabled |
| `--save-every-minutes` | `5.0` | `5.0` | `5.0` |

## 5. 5-Minute Checkpointing & Resume Plan
- `train_uni_v2.py` runs with `--save-every-minutes 5.0`, atomically writing `uni_omr_v2_latest.pt` + `uni_omr_v2_ema_latest.pt` into `/content/vol/ckpts/2026-11-26T01-38-ov-24-mixstyle-domain-invariance/<arm>/`.
- Completed arms persist `arm_result.json`; re-running `run_colab.py` skips completed arms cleanly.

## 6. Evaluation Gate & Decision Tree
- **Metrics Measured (EMA checkpoint):**
  1. `proxy_val.jsonl`: in-domain `printed` vs zero-shot `hw` (Val CE, Val MTP, greedy SER, Token F1, Sys-Count Exact, plus `hw_minus_printed_f1_gap`).
  2. `suite31_hw` (90 pages): `printed_clean`, `printed_photo`, `hw_clean`, `hw_photo` (Token F1, Pitch-only F1, Rhythm-only F1, SER, and cross-domain transfer gap `hw_mean_f1 - printed_mean_f1`).
- **Decision Tree:**
  - **PROMOTE:** If `ms_20` or `ms_40` improves zero-shot `suite31_hw` HW Token F1 (or `proxy_val` HW Token F1) by `>= +0.015` while keeping `printed` Token F1 within `<= 0.01` of `ms_00` -> lock the winning `--mixstyle-prob` as the Iter-11 `UniOMR v2` default (replacing the unablated `0.30` default).
  - **ITERATE:** If `ms_40` improves HW transfer most strongly without saturating printed F1, test `0.40` vs `0.60` or restrict `MixStyle2d` to early FPN stages only (`stage 1-2`).
  - **DISCARD:** If `ms_00` matches or beats both `ms_20` and `ms_40` on both printed and zero-shot HW transfer -> disable `MixStyle2d` (`--mixstyle-prob 0.0`) in Iter-11 `UniOMR v2`.
