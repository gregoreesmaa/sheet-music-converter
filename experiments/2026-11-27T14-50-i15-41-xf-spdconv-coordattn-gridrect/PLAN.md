# PLAN — `2026-11-27T14-50-i15-41-xf-spdconv-coordattn-gridrect`

## 1. Platform & Budget
- **Platform:** Google Colab `1× NVIDIA A100` / `H100` (or Local `RTX 4080 SUPER` 16 GB, native BF16).
- **Estimated Wall-Clock:** ~26 min total on A100 (2 arms × 600 steps + val CE + exact Hybrid INT4 packing & dequant eval).
- **Baseline Commit SHA:** `6055c49`.
- **Self-Containment:** All experimental modules (`SPDConvDown`, `CoordAttn2d`, `VerticalGridRectifier`) live strictly inside `run_colab.py`; `tmp-train/mvd/uni_omr_v2.py` is untouched.

## 2. Identifiers for Follow-Up Queries
- **Experiment ID:** `2026-11-27T14-50-i15-41-xf-spdconv-coordattn-gridrect`
- **Hugging Face (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-27T14-50-i15-41-xf-spdconv-coordattn-gridrect.json`
  - Run Ledger: `runs/2026-11-27T14-50-i15-41-xf-spdconv-coordattn-gridrect.json`
  - Partial Arm Sync: `checkpoints/2026-11-27T14-50-i15-41-xf-spdconv-coordattn-gridrect/{ctrl,spd_coord_rect}_arm_result.json`

## 3. Data & Controlled 2-Arm Protocol
- **Dataset:** `datasets/iter10_5_proxy_2k.tar` (`2,000` train / `64` val full pages, permissive licenses only, `0` `CC-BY-NC`, `2×` dense multi-staff & HW sampling weight, `FULLPAGE_MAX_DIM=1280`, `MAX_SEQ_LEN=1536`, cached in `WORK_ROOT/hf_dl`).
- **Shared Hyperparameters:** `600` steps/arm, `seed=42`, BF16 autocast, `batch_size=2`, `grad_accum=4` (eff. batch 8), **verified `muon+adamw`** (`lr=3e-4`, `weight_decay=0.04`, `no_decay_1d=True`; asserts `opt.mode == "muon+adamw"` on CUDA so no silent AdamW fallback is possible), WSD cosine schedule (`warmup=60`, `decay_frac=0.25`, `min_lr_ratio=0.05`), EMA (`decay=0.99` every step), `QATInt4ControllerV2` enabled at step `420` (final 30%), `grad_checkpoint=True`.
- **Windows RTX 4080 SUPER Safety:** Automatically disables `torch.backends.cudnn.enabled` and forces CUTLASS `EFFICIENT_ATTENTION` on `win32`.
- **Arm 0 (`ctrl`):** `UniOMRv2Config.edge85m(n_axial_layers=0, n_global_layers=6, grad_checkpoint=True)` with stock `AnisotropicMusicBackbone`.
- **Arm 1 (`spd_coord_rect`):** Same `edge85m + ax0_gl6` config, with:
  1. `SPDConvDown` replacing strided `stage1.down` (`/1→/2`) and `stage2.down` (`/2→/4`) via `PixelUnshuffle(2)` + non-strided `3×3` conv + `GroupNorm` + `SiLU`.
  2. `CoordAttn2d` (factorized 1D horizontal + vertical strip pooling + shared bottleneck) appended after `stage2`, `stage3`, and `stage4`.
  3. `VerticalGridRectifier` (identity-initialized residual 1D vertical displacement field `dy(y, x)` bounded to `±0.12`, applied via `F.grid_sample` on the `/8` backbone output before the ViT encoder, regularized with a spatial smoothness penalty `weight=0.05`).

## 4. Verification & Launch Commands
CPU 2-step smoke test (local, zero GPU):
```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe -m py_compile experiments\2026-11-27T14-50-i15-41-xf-spdconv-coordattn-gridrect\run_colab.py
.\.venv\Scripts\python.exe experiments\2026-11-27T14-50-i15-41-xf-spdconv-coordattn-gridrect\run_colab.py --smoke-test
```
Colab A100 / Local 4080 SUPER run:
```bash
python3 -u experiments/2026-11-27T14-50-i15-41-xf-spdconv-coordattn-gridrect/run_colab.py
```

## 5. Checkpointing & Remote HF Auto-Resume Plan
- Atomic checkpoint save (`latest.pt.tmp` → `os.replace`) every `5.0` minutes per arm under `WORK_ROOT/ckpts/<arm>/` with full step/optimizer/EMA/RNG state for mid-arm local resume.
- **1-Arm HF Partial Sync & Startup Auto-Resume:** On startup, checks HF Private Hub for `evals/<EXP_ID>.json` and `checkpoints/<EXP_ID>/{ctrl,spd_coord_rect}_arm_result.json` and skips any already-completed arm. After Arm 0 (`ctrl`) finishes, immediately uploads both `checkpoints/<EXP_ID>/ctrl_arm_result.json` and partial `evals/<EXP_ID>.json` (`status: "RUNNING"`) to HF before starting Arm 1 (`spd_coord_rect`).
- Security: reads `HF_TOKEN` from `/tmp/.hf_token` (chmod 600), env `HF_TOKEN`, or `~/.cache/huggingface/token`, never prints it, always unlinks `/tmp/.hf_token` in `finally`.

## 6. Evaluation Gate & Decision Tree
- **Primary Metrics:** `ema_val_ce`, `int4_dequant_val_ce`, `step_ms`, and exact serialized `packed_int4_mb` (`< 55.0 MB` hard budget).
- **PROMOTE:** `spd_coord_rect` beats `ctrl` by `>= -0.015` on `ema_val_ce` (and `int4_dequant_val_ce`), stays `< 55.0 MB` packed INT4, and `step_ms` overhead is `<= +12%` → port into `AnisotropicMusicBackbone` in `uni_omr_v2.py` for Iter 15 pretraining.
- **ITERATE:** Beats `ctrl` by `-0.005` to `-0.015` or shows high latency from one component → run component-level ablation (`SPD-Conv` only vs `CoordAttn2d` only vs `VerticalGridRectifier` only) to isolate the winning sub-module.
- **DISCARD:** `< -0.005` gain on `ema_val_ce` over `ctrl` at 600 steps (or unstable warp / INT4 degradation `> +0.025`) → keep stock `AnisotropicMusicBackbone`.
