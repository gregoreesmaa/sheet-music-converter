# Experiment Plan — `2026-11-26T00-55-ablation-spinquant-int4-parity`

## 1. Platform Chosen & Rationale

- **Platform:** **Modal `L4` (`24 GB`, `cc=8.9`)** with `32 GB` RAM and `4 vCPU` (`~$0.80/hr`).
- **Rationale:** Full-size `small_45m` + `edge85m` FP32/INT4 parity across 12 real `1024px` `suite31_hw` pages runs cleanly in `24 GB` VRAM without touching the local RTX 4080 SUPER, while `32 GB` host RAM gives generous headroom for full-size `edge85m` multi-target ONNX export (`webgpu`, `npu`, `cpu`) and `MatMulNBitsQuantizer`.
- **Estimated Wall-Clock & Cost:** `~10–15 min` total (`~$0.15–$0.20` of `$30/mo` free credit).

## 2. Identifiers for Follow-Up Queries (Zero-Prior-Context Handoff)

- **Modal Profile:** `gregoreesmaa1`
- **Modal App Name:** `exp8-ablation-spinquant-int4-parity`
- **Modal App ID:** `ap-OrTAfTJlPV0o9aSWO0r59l`
- **Modal Function Call ID:** `fc-01M8AFE2AHDQ7KS9W8JQX0T4PT`
- **Modal Dashboard URL:** `https://modal.com/apps/gregoreesmaa1/main/ap-OrTAfTJlPV0o9aSWO0r59l`
- **Modal Volume:** `omr-cache` mounted at `/vol` (cached eval tar at `/vol/datasets/suite31_hw_eval/`, output JSON at `/vol/evals/2026-11-26T00-55-ablation-spinquant-int4-parity.json`)
- **Modal Secret:** `hf-secret` (`HF_TOKEN`)
- **Hugging Face Repo:** `gregoreesmaa/sheet-music-converter-omr` (`repo_type="dataset"`)
  - Eval Report: `evals/2026-11-26T00-55-ablation-spinquant-int4-parity.json`
  - Run Ledger: `runs/2026-11-26T00-55-ablation-spinquant-int4-parity.json`

### Copy-Paste Monitoring / Retrieval Commands

```powershell
# Check app status
.\.venv\Scripts\modal.exe app list

# Stream live logs
.\.venv\Scripts\modal.exe app logs ap-OrTAfTJlPV0o9aSWO0r59l

# Stop runaway job if needed
.\.venv\Scripts\modal.exe app stop ap-OrTAfTJlPV0o9aSWO0r59l

# Download completed eval JSON from Hugging Face
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('gregoreesmaa/sheet-music-converter-omr', 'evals/2026-11-26T00-55-ablation-spinquant-int4-parity.json', repo_type='dataset'))"
```

## 3. Required Data & Models (100% Permissive)

- **Eval Images:** `datasets/suite31_hw_eval.tar` on `gregoreesmaa/sheet-music-converter-omr` (45-piece `suite31_hw` benchmark; 12 deterministic pages selected: 3 `printed_clean`, 3 `printed_photo`, 3 `hw_clean`, 3 `hw_photo`, resized aspect-preserving to `max_dim=1024`, H/W multiples of 16).
- **Models Benchmarked (Commit `9d73170`):**
  - `UniOMRv2Config.small_45m()` (`~41.3M` params, `d_model=384`, `8Q/2KV`, 2 axial + 4 global + 6 dec layers, MTP-4)
  - `UniOMRv2Config.edge85m()` (`~78.1M` params, `d_model=512`, `8Q/2KV`, 2 axial + 6 global + 6 dec layers, MTP-4)
- **Weight Regimes Tested:**
  - **Clean weights:** deterministic `seed=42` `UniOMRv2Net` weights.
  - **Outlier-stressed weights:** deterministic `seed=1234` realistic channel-wise outliers (`2%` of residual channels scaled `12×` in decoder `RMSNorm.weight` + `12× std` spikes on `dec_layers` linear weights, `lm_head`, and encoder `V/O` head channels).

## 4. Exact Launch Command

```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\modal.exe run --detach experiments/2026-11-26T00-55-ablation-spinquant-int4-parity/run_modal.py
```

## 5. Checkpointing & Recovery Plan

- This is a short (~10–15 min) deterministic benchmark rather than a multi-hour training run; no model training checkpoints are produced.
- Extracted `suite31_hw_eval.tar` is persisted on shared Modal Volume `omr-cache` (`/vol/datasets/suite31_hw_eval/`) with `vol.commit()` so re-runs skip downloading/extracting.
- On completion, the full benchmark JSON is committed to `/vol/evals/2026-11-26T00-55-ablation-spinquant-int4-parity.json` AND uploaded to HF `evals/2026-11-26T00-55-ablation-spinquant-int4-parity.json` + `runs/2026-11-26T00-55-ablation-spinquant-int4-parity.json`.
- If preempted, simply re-run the exact launch command above (idempotent, reuses cached `/vol/datasets/suite31_hw_eval/`).

## 6. Evaluation Gate & Decision Tree ("What We Do With the Result")

- **Metrics Measured:**
  1. `fp32_max_abs_diff_worst` & `fp32_mse_mean` (SpinQuant invariance gate)
  2. INT4 (`block_size=32`) `orig_int4_mse` vs `folded_int4_mse`, `orig_int4_kl` vs `folded_int4_kl`, `orig_int4_top1_agree` vs `folded_int4_top1_agree` on both clean and outlier-stressed weights
  3. Multi-target ONNX sizes (`webgpu`, `npu`, `cpu`) for `small_45m` and `edge85m`
- **Promote:**
  - `fp32_max_abs_diff_worst < 1e-3` on both presets (clean & outlier-stressed), AND
  - Outlier-stressed INT4 MSE & KL drop by `>= 2×` with higher top-1 agreement after `fold_hadamard_v2`, AND
  - `webgpu` total ONNX size `< 55 MB` for both `small_45m` and `edge85m`.
  - **Action:** Wire `fold_hadamard_v2` as default pre-export step before `export_uni_v2_onnx.py` for the upcoming Iter-11 `UniOMR v2` WebGPU/NPU/CPU release!
- **Iterate:**
  - If `edge85m` WebGPU total exceeds `55 MB` or FP32 invariance shows unexpected drift on any layer → audit which layer/initializer inflated size or drift (`tok_emb` / `lm_head` / `Conv` compression) and fix in `export_uni_v2_onnx.py` or `spinquant_fold.py`.
- **Discard:**
  - If `fold_hadamard_v2` degrades INT4 parity on both clean and outlier-stressed weights → keep standard unrotated `QATInt4ControllerV2` export path.
