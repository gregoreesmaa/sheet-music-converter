# Experiment Plan — `2026-11-26T01-38-ov-42-v1-to-v2-distill-temp-jsd`

> **Zero-Prior-Context Handoff:** Everything needed to run, monitor, stop, resume, or evaluate Overnight Exp 42 (`v1 → v2` distillation temperature × JSD/KL × alpha sweep) is self-contained in this folder (`README.md`, `PLAN.md`, `run_colab.py`).

## 1. Platform Chosen & Rationale
- **Provider:** Google Colab (`google-colab-cli` via `.\.venv\Scripts\colab.exe`).
- **Hardware:** `1x NVIDIA L4` (`24 GB` VRAM, Ada Lovelace `cc=8.9`, **native BF16**, `torch 2.14.1+cu130` pre-installed).
- **Session Name:** `ov-42-v1-v2-kd`
- **Estimated Wall-Clock & Cost:** ~35–42 minutes total (9 arms × 350 steps × ~3.8 min/arm including frozen BF16 `v1` teacher forward + `proxy_val` eval) ≈ `~3.0–3.5 CU`.
- **Rationale:** 24 GB VRAM comfortably holds the frozen `UniOMR v1` teacher + `UniOMR v2 small_45m` student + EMA in native BF16 without OOM, and the 2-stage design covers all `T ∈ {1.0, 2.0, 4.0}`, `{G-JSD, FKL}`, and `alpha ∈ {0.0, 0.25, 0.50, 0.75}` inside one ~40-minute session.

## 2. Identifiers for Follow-Up Queries
- **Colab Session:** `ov-42-v1-v2-kd`
- **Hugging Face Repo:** `gregoreesmaa/sheet-music-converter-omr` (`repo_type="dataset"`)
  - Frozen Teacher Ckpt: `checkpoints/iter10_soup_baseline/uni_omr_soup.pt`
  - Full Eval JSON: `evals/2026-11-26T01-38-ov-42-v1-to-v2-distill-temp-jsd.json`
  - Run Ledger JSON: `runs/2026-11-26T01-38-ov-42-v1-to-v2-distill-temp-jsd.json`
- **Remote Paths on Colab VM:**
  - Code bundle: `/content/ov_bundle.tar.gz` → extracted to `/workspace/`
  - Datasets & rolling ckpts: `/content/vol/`
  - Transient token file: `/tmp/.hf_token` (`chmod 600`, purged on read and in `finally`)

### Copy-Paste Monitoring & Control Commands (PowerShell)
```powershell
# Check active Colab sessions & compute-unit balance
.\.venv\Scripts\colab.exe sessions
.\.venv\Scripts\colab.exe usage

# Stop Colab session immediately (ALWAYS run in finally!)
.\.venv\Scripts\colab.exe stop -s ov-42-v1-v2-kd

# Fetch completed eval JSON from Hugging Face
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; p = hf_hub_download('gregoreesmaa/sheet-music-converter-omr', 'evals/2026-11-26T01-38-ov-42-v1-to-v2-distill-temp-jsd.json', repo_type='dataset'); print(open(p).read())"
```

## 3. Required Training & Eval Data
- **Supervised Proxy Dataset:** `datasets/iter10_5_proxy_2k.tar` (`2,000` train + `64` val full pages, permissive only, 0 `suite31_hw` overlap).
- **Teacher Checkpoint:** `checkpoints/iter10_soup_baseline/uni_omr_soup.pt` (frozen BF16 `UniOMRNet`). Automatic vocab alignment handles both matched vocab and legacy 182→237 vocab cleanly.
- **Student Initialization:** Scratch `UniOMRv2Net(UniOMRv2Config.small_45m(grad_checkpoint=True))` with identical `seed=42` and identical batch stream across all 9 arms.

## 4. Exact Launch Command & Hyperparameters
```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\colab.exe new -s ov-42-v1-v2-kd --gpu L4
.\.venv\Scripts\colab.exe upload -s ov-42-v1-v2-kd <local_bundle.tar.gz> /content/ov_bundle.tar.gz
.\.venv\Scripts\colab.exe upload -s ov-42-v1-v2-kd <local_tmp_token> /tmp/.hf_token
.\.venv\Scripts\colab.exe exec -s ov-42-v1-v2-kd -f experiments/2026-11-26T01-38-ov-42-v1-to-v2-distill-temp-jsd/run_colab.py
.\.venv\Scripts\colab.exe stop -s ov-42-v1-v2-kd
```

| Stage | Arm | `kd_temp (T)` | `kd_beta` (Objective) | `kd_alpha` | Steps |
|---|---|---:|---|---:|---:|
| Control | `ctrl_no_kd` | — | — | `0.00` | `350` |
| Stage 1 | `t1.0_jsd_a0.50` | `1.0` | `0.5` (G-JSD) | `0.50` | `350` |
| Stage 1 | `t2.0_jsd_a0.50` | `2.0` | `0.5` (G-JSD) | `0.50` | `350` |
| Stage 1 | `t4.0_jsd_a0.50` | `4.0` | `0.5` (G-JSD) | `0.50` | `350` |
| Stage 1 | `t1.0_fkl_a0.50` | `1.0` | `1.0` (Forward KL) | `0.50` | `350` |
| Stage 1 | `t2.0_fkl_a0.50` | `2.0` | `1.0` (Forward KL) | `0.50` | `350` |
| Stage 1 | `t4.0_fkl_a0.50` | `4.0` | `1.0` (Forward KL) | `0.50` | `350` |
| Stage 2 | `best_a0.25` | `T*` | `beta*` | `0.25` | `350` |
| Stage 2 | `best_a0.75` | `T*` | `beta*` | `0.75` | `350` |

## 5. 5-Minute Checkpointing & Resume Plan
- Each arm saves 5-minute atomic checkpoints to `/content/vol/ckpts/<exp_id>/<arm>/` and caches its completed `arm_result.json`, so re-running skips finished arms cleanly.

## 6. Evaluation Gate & Decision Tree
- **Primary Metrics (on 64-page `proxy_val`):**
  1. `val_token_f1` & `delta_val_tok_f1_vs_ctrl`
  2. `val_ce_loss` & `delta_val_ce_vs_ctrl`
  3. `val_ser`, `val_sys_count_exact`
  4. `mean_kd_loss`, `mean_step_ms`, `peak_vram_gb`
- **Decision Tree:**
  - **PROMOTE:** Winning `(T*, beta*, alpha*)` beats `ctrl_no_kd` by `>= +0.015` `val_token_f1` (or `<= -0.05` `val_ce_loss`) → set `--kd-temp T* --kd-beta beta* --kd-alpha alpha*` as the default Iter-11 `v1 → v2` distillation hyperparameters in `train_uni_v2.py`.
  - **ITERATE:** KD lowers `val_ce_loss` strongly but slows throughput by `> 1.5x` → switch from online `v1` teacher to precomputed `TopKTeacherCache` (`K=8`) using the winning `(T*, beta*, alpha*)`.
  - **DISCARD:** All 8 KD arms fail to beat `ctrl_no_kd` → train Iter-11 `UniOMR v2` purely supervised from `I-JEPA` warm-start.
