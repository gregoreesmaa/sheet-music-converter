# PLAN — `2026-11-27T14-50-i15-43-xf-mrfp-and-delta-debias`

## 1. Platform & Budget
- **Platform:** Google Colab `1× NVIDIA A100` / `H100` (or Local `RTX 4080 SUPER` 16 GB, native BF16).
- **Estimated Wall-Clock:** ~26 min total on A100 (2 arms × 600 steps + paired clean/degraded val CE + DELTA SVD subspace sweep + exact Hybrid INT4 packing & dequant eval).
- **Baseline Commit SHA:** `6055c49`.
- **Self-Containment:** All `MRFP` and `DELTA` logic lives strictly inside `run_colab.py`; `tmp-train/mvd/uni_omr_v2.py` is untouched.

## 2. Identifiers for Follow-Up Queries
- **Experiment ID:** `2026-11-27T14-50-i15-43-xf-mrfp-and-delta-debias`
- **Short ID:** `i15-43` (`wave=0`, P0 cross-field)
- **Hugging Face (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-27T14-50-i15-43-xf-mrfp-and-delta-debias.json`
  - Run Ledger: `runs/2026-11-27T14-50-i15-43-xf-mrfp-and-delta-debias.json`
  - Partial Arm Sync: `checkpoints/2026-11-27T14-50-i15-43-xf-mrfp-and-delta-debias/{ctrl,mrfp_delta}_arm_result.json`

## 3. Data & Controlled 2-Arm Protocol
- **Dataset:** `datasets/iter10_5_proxy_2k.tar` (`2,000` train / `64` val full pages, permissive licenses only, `0` `CC-BY-NC`, `2×` dense multi-staff & HW sampling weight, `FULLPAGE_MAX_DIM=1280`, `MAX_SEQ_LEN=1536`).
- **Paired Degraded Validation Split:** Deterministic in-memory phone-photo/scan degradation applied to the 64 val pages (contrast/gamma jitter, uneven illumination ramp, sepia/paper tint, mild Gaussian blur, sensor noise) for paired `DELTA` nuisance-subspace estimation (`first 32` paired val crops) and held-out clean + degraded evaluation (`all 64` val pages + held-out `last 32` val pages).
- **Shared Hyperparameters:** `600` steps/arm, `seed=42`, BF16 autocast, `batch_size=2`, `grad_accum=4` (eff. batch 8), **verified `muon+adamw`** (`lr=3e-4`, `weight_decay=0.04`, `no_decay_1d=True`; asserts `opt.mode == "muon+adamw"` on CUDA), WSD cosine schedule (`warmup=60`, `decay_frac=0.25`, `min_lr_ratio=0.05`), EMA (`decay=0.99` every step), `QATInt4ControllerV2` enabled at step `420` (final 30%), `grad_checkpoint=True`.
- **Windows RTX 4080 SUPER Safety:** Disables `torch.backends.cudnn.enabled` and forces CUTLASS `EFFICIENT_ATTENTION` on `win32`.
- **Arm 0 (`ctrl`):** `UniOMRv2Config.edge85m(n_axial_layers=0, n_global_layers=6, grad_checkpoint=True)` trained cleanly, then evaluated pre-DELTA and post-DELTA on clean and degraded val splits.
- **Arm 1 (`mrfp_delta`):** Identical `edge85m + ax0_gl6` architecture (`+0.000 MB` INT4), with parameter-free `MRFP` active across `stage1/stage2` (fine-scale channel stats randomization + batch style mix) and `stage3/stage4` (coarse-scale style & low-frequency spatial envelope perturbation) during training, linearly annealed `1.0 -> 0.0` over steps `420..600`. Post-training, estimates the top-`k` (`k ∈ {2, 4, 8, 16}`) synthetic-appearance nuisance subspace via `DELTA` SVD on paired clean-vs-degraded val features and folds `P_k = I - U_k U_k^T` into decoder cross-attention `k_proj`/`v_proj` weights.

## 4. Verification & Launch Commands
CPU 2-step smoke test (local, zero GPU):
```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe experiments\2026-11-27T14-50-i15-43-xf-mrfp-and-delta-debias\run_colab.py --smoke-test
```
Colab A100 / Local 4080 SUPER run:
```bash
python3 -u experiments/2026-11-27T14-50-i15-43-xf-mrfp-and-delta-debias/run_colab.py
```

## 5. Checkpointing & Remote HF Auto-Resume Plan
- Atomic checkpoint save (`latest.pt.tmp` → `os.replace`) every `5.0` minutes per arm under `WORK_ROOT/ckpts/<arm>/` with full step/optimizer/EMA/RNG state.
- **1-Arm HF Partial Sync & Startup Auto-Resume:** On startup, checks HF Private Hub for `evals/<EXP_ID>.json` and `checkpoints/<EXP_ID>/{ctrl,mrfp_delta}_arm_result.json` and skips any already-completed arm. Uploads Arm 0 (`ctrl`) JSON + partial `evals/<EXP_ID>.json` (`status: "RUNNING"`) immediately before starting Arm 1 (`mrfp_delta`).
- Security: reads `HF_TOKEN` from `/tmp/.hf_token` (chmod 600), env `HF_TOKEN`, or `~/.cache/huggingface/token`, never prints it, always unlinks `/tmp/.hf_token` in `finally`.

## 6. Evaluation Gate & Decision Tree
- **Primary Metrics:** `ema_val_ce` (clean), `ema_val_ce_degraded`, `delta_val_ce_clean`, `delta_val_ce_degraded`, `int4_dequant_val_ce`, `int4_dequant_val_ce_degraded`, `step_ms`, and `packed_int4_mb` (`< 55.0 MB`, exact `+0.000 MB` overhead).
- **PROMOTE:** `mrfp_delta` improves degraded val CE by `>= -0.020` (and clean val CE `<= +0.005` or improved) with `packed_int4_mb < 55.0 MB` and `step_ms` overhead `<= +8%` → integrate `MRFP` into `AnisotropicMusicBackbone` training loop and add post-training `DELTA` subspace folding utility for Iter 15.
- **ITERATE:** `MRFP` helps degraded val CE but `DELTA` over-projects clean features (or vice versa) → separate `MRFP`-only vs. `DELTA`-only with soft shrinkage `I - alpha * U_k U_k^T` (`alpha ∈ {0.5, 0.75}`).
- **DISCARD:** `< -0.005` gain on both clean and degraded val CE over `ctrl` → drop cleanly.
