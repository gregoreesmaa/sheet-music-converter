# PLAN — `2026-11-27T14-50-i15-42-xf-aux-cbdice-unmix-gdcm`

## 1. Platform & Budget
- **Platform:** Google Colab `1× NVIDIA A100` / `H100` (or Local `RTX 4080 SUPER` 16 GB, native BF16).
- **Estimated Wall-Clock:** ~25 min total on A100 (2 arms × 600 steps + val CE + exact Hybrid INT4 packing & dequant eval).
- **Baseline Commit SHA:** `6055c49`.
- **Self-Containment:** All experimental training-only heads and losses (`InkUnmixHead`, `GDCMCountHead`, `compute_cbdice_staff_loss`, `compute_ink_unmix_loss`, `compute_gdcm_count_loss`) live strictly inside `run_colab.py`; `tmp-train/mvd/uni_omr_v2.py` is untouched.

## 2. Identifiers for Follow-Up Queries
- **Experiment ID:** `2026-11-27T14-50-i15-42-xf-aux-cbdice-unmix-gdcm`
- **Hugging Face (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-27T14-50-i15-42-xf-aux-cbdice-unmix-gdcm.json`
  - Run Ledger: `runs/2026-11-27T14-50-i15-42-xf-aux-cbdice-unmix-gdcm.json`
  - Partial Arm Sync: `checkpoints/2026-11-27T14-50-i15-42-xf-aux-cbdice-unmix-gdcm/{ctrl,aux_topo_unmix_gdcm}_arm_result.json`

## 3. Data & Controlled 2-Arm Protocol
- **Dataset:** `datasets/iter10_5_proxy_2k.tar` (`2,000` train / `64` val full pages, permissive licenses only, `0` `CC-BY-NC`, `2×` dense multi-staff & HW sampling weight, `FULLPAGE_MAX_DIM=1280`, `MAX_SEQ_LEN=1536`, cached in `WORK_ROOT/hf_dl`).
- **Shared Hyperparameters:** `600` steps/arm, `seed=42`, BF16 autocast, `batch_size=2`, `grad_accum=4` (eff. batch 8), **verified `muon+adamw`** (`lr=3e-4`, `weight_decay=0.04`, `no_decay_1d=True`; asserts `opt.mode == "muon+adamw"` on CUDA so no silent AdamW fallback is possible), WSD cosine schedule (`warmup=60`, `decay_frac=0.25`, `min_lr_ratio=0.05`), EMA (`decay=0.99` every step), `QATInt4ControllerV2` enabled at step `420` (final 30%), `grad_checkpoint=True`.
- **Windows RTX 4080 SUPER Safety:** Automatically disables `torch.backends.cudnn.enabled` and forces CUTLASS `EFFICIENT_ATTENTION` on `win32`.
- **Arm 0 (`ctrl`):** `UniOMRv2Config.edge85m(n_axial_layers=0, n_global_layers=6, grad_checkpoint=True)` trained with standard CE + MTP-4 (`0.25`) + router saliency (`0.10`).
- **Arm 1 (`aux_topo_unmix_gdcm`):** Identical inference architecture, adding 3 **training-only** auxiliary heads stripped cleanly by `strip_aux_heads()` (**exact `0.000 MB` INT4 delta**, **0% inference cost**):
  1. `InkUnmixHead` on `P2 (/4)` predicting 2 channels (primary foreground ink vs background/bleed-through), supervised on self-derived high-pass ink/paper masks (`weight=0.10`).
  2. `cbDice` soft-skeleton thin-line preservation loss (3-iter differentiable morphological skeletonization + BCE/Dice) on the existing `StaffSegHead` (`P2+P3`) output (`weight=0.10`).
  3. `GDCMCountHead` (Global Dynamic Counting Module) predicting 6 per-page symbol/structure counts (`systems`, `barlines`, `notes`, `rests`, `accidentals`, `dots/ties`) from pooled `mem_final` via `SmoothL1Loss` on `log1p(count)` (`weight=0.10`).

## 4. Verification & Launch Commands
CPU 2-step smoke test (local, zero GPU):
```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe -m py_compile experiments\2026-11-27T14-50-i15-42-xf-aux-cbdice-unmix-gdcm\run_colab.py
.\.venv\Scripts\python.exe experiments\2026-11-27T14-50-i15-42-xf-aux-cbdice-unmix-gdcm\run_colab.py --smoke-test
```
Colab A100 / Local 4080 SUPER run:
```bash
python3 -u experiments/2026-11-27T14-50-i15-42-xf-aux-cbdice-unmix-gdcm/run_colab.py
```

## 5. Checkpointing & Remote HF Auto-Resume Plan
- Atomic checkpoint save (`latest.pt.tmp` → `os.replace`) every `5.0` minutes per arm under `WORK_ROOT/ckpts/<arm>/` with full step/optimizer/EMA/RNG state for mid-arm local resume.
- **1-Arm HF Partial Sync & Startup Auto-Resume:** On startup, checks HF Private Hub for `evals/<EXP_ID>.json` and `checkpoints/<EXP_ID>/{ctrl,aux_topo_unmix_gdcm}_arm_result.json` and skips any already-completed arm. After Arm 0 (`ctrl`) finishes, immediately uploads both `checkpoints/<EXP_ID>/ctrl_arm_result.json` and partial `evals/<EXP_ID>.json` (`status: "RUNNING"`) to HF before starting Arm 1 (`aux_topo_unmix_gdcm`).
- Security: reads `HF_TOKEN` from `/tmp/.hf_token` (chmod 600), env `HF_TOKEN`, or `~/.cache/huggingface/token`, never prints it, always unlinks `/tmp/.hf_token` in `finally`.

## 6. Evaluation Gate & Decision Tree
- **Primary Metrics:** `ema_val_ce`, `int4_dequant_val_ce`, `step_ms`, and exact serialized `packed_int4_mb` (must verify **exact `0.000 MB` stripped INT4 delta** vs `ctrl`).
- **PROMOTE:** `aux_topo_unmix_gdcm` beats `ctrl` by `>= -0.010` on `ema_val_ce` (and `int4_dequant_val_ce`) with exact `0.000 MB` stripped INT4 delta → port all 3 training-only heads/losses into `uni_omr_v2.py` + `train_uni_v2.py` for Iter 15 pretraining (free accuracy boost at 0% WebGPU inference cost!).
- **ITERATE:** Beats `ctrl` by `-0.003` to `-0.010` or one aux loss dominates gradients → tune loss weights or ablate `InkUnmixHead` / `cbDice` / `GDCM` individually.
- **DISCARD:** `< -0.003` gain on `ema_val_ce` over `ctrl` at 600 steps (or slows training `> +20%` without CE benefit) → drop the extra training-only heads.
