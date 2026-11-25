# Experiment Plan — `2026-11-26T01-38-ov-41-qat-int4-blocksize-spinquant`

> **Zero-Prior-Context Handoff:** Everything needed to run, monitor, stop, resume, or evaluate Overnight Exp 41 (QAT-INT4 `block_size` × SpinQuant sweep) is self-contained in this folder (`README.md`, `PLAN.md`, `run_colab.py`).

## 1. Platform Chosen & Rationale
- **Provider:** Google Colab (`google-colab-cli` via `.\.venv\Scripts\colab.exe`).
- **Hardware:** `1x NVIDIA L4` (`24 GB` VRAM, Ada Lovelace `cc=8.9`, **native BF16**, `torch 2.14.1+cu130` pre-installed).
- **Session Name:** `ov-41-qat-spinquant`
- **Estimated Wall-Clock & Cost:** ~32–38 minutes total (~7 min shared 600-step base + 6 × ~3.5 min 200-step QAT branches + ~5 min PTQ/QAT logit-KL & `proxy_val` evals) ≈ `~2.8–3.2 CU`.
- **Rationale:** Branching all 6 conditions from a single trained base checkpoint keeps the entire 3 × 2 factorial sweep cleanly inside a single ~35-minute L4 run with zero initialization noise.

## 2. Identifiers for Follow-Up Queries
- **Colab Session:** `ov-41-qat-spinquant`
- **Hugging Face Repo:** `gregoreesmaa/sheet-music-converter-omr` (`repo_type="dataset"`)
  - Full Eval JSON: `evals/2026-11-26T01-38-ov-41-qat-int4-blocksize-spinquant.json`
  - Run Ledger JSON: `runs/2026-11-26T01-38-ov-41-qat-int4-blocksize-spinquant.json`
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
.\.venv\Scripts\colab.exe stop -s ov-41-qat-spinquant

# Fetch completed eval JSON from Hugging Face
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; p = hf_hub_download('gregoreesmaa/sheet-music-converter-omr', 'evals/2026-11-26T01-38-ov-41-qat-int4-blocksize-spinquant.json', repo_type='dataset'); print(open(p).read())"
```

## 3. Required Training & Eval Data
- **Supervised Proxy Dataset:** `datasets/iter10_5_proxy_2k.tar` (`2,000` train + `64` val full pages, permissive only, 0 `suite31_hw` overlap).
- **Initialization:** Shared `UniOMRv2Net(UniOMRv2Config.small_45m(grad_checkpoint=True))` trained for `600` steps (`seed=42`, `lr=3e-4`, `B=2`, `grad_accum=4`, `fullpage_max_dim=1280`).

## 4. Exact Launch Command & Hyperparameters
```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\colab.exe new -s ov-41-qat-spinquant --gpu L4
.\.venv\Scripts\colab.exe upload -s ov-41-qat-spinquant <local_bundle.tar.gz> /content/ov_bundle.tar.gz
.\.venv\Scripts\colab.exe upload -s ov-41-qat-spinquant <local_tmp_token> /tmp/.hf_token
.\.venv\Scripts\colab.exe exec -s ov-41-qat-spinquant -f experiments/2026-11-26T01-38-ov-41-qat-int4-blocksize-spinquant/run_colab.py
.\.venv\Scripts\colab.exe stop -s ov-41-qat-spinquant
```

| Arm | `block_size` | `SpinQuant` (`fold_hadamard_v2`) | QAT Steps | QAT LR |
|---|---:|---|---:|---:|
| `bs32_no_sq` | `32` | Off | `200` | `1e-4` cosine → `1e-5` |
| `bs32_sq` | `32` | On (R1 + R2 before QAT) | `200` | `1e-4` cosine → `1e-5` |
| `bs64_no_sq` | `64` | Off | `200` | `1e-4` cosine → `1e-5` |
| `bs64_sq` | `64` | On (R1 + R2 before QAT) | `200` | `1e-4` cosine → `1e-5` |
| `bs128_no_sq` | `128` | Off | `200` | `1e-4` cosine → `1e-5` |
| `bs128_sq` | `128` | On (R1 + R2 before QAT) | `200` | `1e-4` cosine → `1e-5` |

## 5. 5-Minute Checkpointing & Resume Plan
- Shared base checkpoint is saved atomically to `/content/vol/ckpts/<exp_id>/base_fp_600.pt` (and every `5.0` min during base training).
- Each QAT arm caches its completed result to `/content/vol/ckpts/<exp_id>/<arm>/arm_result.json` and checkpoints every `5.0` min, so re-running skips completed stages cleanly.

## 6. Evaluation Gate & Decision Tree
- **Primary Metrics (per arm, both 0-step PTQ and 200-step QAT):**
  1. `val_token_f1` & `f1_retention_vs_fp32` (`qat_val_f1 / fp32_ref_val_f1`)
  2. `logit_kl_vs_fp32` (`KL(P_fp32 || P_int4)`), `logit_mse_vs_fp32`, `top1_token_agree`
  3. `val_ce_loss`, `val_ser`
  4. Exact WebGPU size: `hybrid_fp16_int4_with_scales_mb` & `pure_int4_with_scales_mb`
  5. SpinQuant FP32 invariance sanity check (`fp32_fold_max_abs_diff`)
- **Decision Tree:**
  - **PROMOTE:** Winning `(block_size*, spinquant*)` achieves `>= 98.5%` F1 retention and lowest `logit_kl_vs_fp32` within `< 55 MB` → set as default Iter-11 QAT-INT4 + ONNX export recipe (`train_uni_v2.py` + `export_uni_v2_onnx.py`).
  - **ITERATE:** SpinQuant helps PTQ dramatically (`>= 30%` KL drop) but QAT partially erodes rotation alignment → test post-QAT Hadamard folding vs pre-QAT folding.
  - **DISCARD:** SpinQuant shows `< 5%` KL improvement across all block sizes → keep plain `QATInt4ControllerV2(block_size=32)`.
