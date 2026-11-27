# PLAN — `2026-11-27T14-55-i15-50-xf-remora-modifiers-daq-fape`

## 1. Platform & Budget
- **Platform:** Google Colab `1× NVIDIA A100` (or Local `RTX 4080 SUPER` 16 GB, native BF16).
- **Estimated Wall-Clock:** ~27 min total on A100 (2 arms × 600 steps + val CE + modifier-token CE + DAQ register-shift detection AUROC/accuracy + exact Hybrid INT4 packing & dequant eval).
- **Baseline Commit SHA:** `6055c49`.
- **Self-Containment:** All `RemoraModifierHead`, `FAPE2DOffsetHead`, and `DAQRegisterCriticHead` code lives strictly inside `run_colab.py`; `tmp-train/mvd/uni_omr_v2.py` is untouched.

## 2. Identifiers for Follow-Up Queries
- **Experiment ID:** `2026-11-27T14-55-i15-50-xf-remora-modifiers-daq-fape`
- **Hugging Face (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-27T14-55-i15-50-xf-remora-modifiers-daq-fape.json`
  - Run Ledger: `runs/2026-11-27T14-55-i15-50-xf-remora-modifiers-daq-fape.json`
  - Partial Arm Sync: `checkpoints/2026-11-27T14-55-i15-50-xf-remora-modifiers-daq-fape/{ctrl,remora_daq_fape}_arm_result.json`

## 3. Data & Controlled 2-Arm Protocol
- **Dataset:** `datasets/iter10_5_proxy_2k.tar` (`2,000` train / `64` val full pages, permissive licenses only, `0` `CC-BY-NC`, `2×` dense multi-staff & HW sampling weight, `FULLPAGE_MAX_DIM=1280`, `MAX_SEQ_LEN=1536`).
- **Shared Hyperparameters:** `600` steps/arm, `seed=42`, BF16 autocast, `batch_size=2`, `grad_accum=4` (eff. batch 8), **verified `muon+adamw`** (`lr=3e-4`, `weight_decay=0.04`, `no_decay_1d=True`; asserts `opt.mode == "muon+adamw"` on CUDA), WSD cosine schedule (`warmup=60`, `decay_frac=0.25`, `min_lr_ratio=0.05`), EMA (`decay=0.99` every step), `QATInt4ControllerV2` enabled at step `420` (final 30%), `grad_checkpoint=True`.
- **Windows RTX 4080 SUPER Safety:** Automatically disables `torch.backends.cudnn.enabled` and forces CUTLASS `EFFICIENT_ATTENTION` on `win32`.
- **Arm 0 (`ctrl`):** Stock `UniOMRv2Config.edge85m(n_axial_layers=0, n_global_layers=6, grad_checkpoint=True)`.
- **Arm 1 (`remora_daq_fape`):**
  - **(a) `RemoraModifierHead` (`~0.3 MB` INT4):** Local cross-attention (`d_mod=128`, `n_heads=4`) from decoder hidden states `h` to `mem_final`, projecting to a gated residual logit delta strictly masked to rare modifier/accidental/dot/articulation vocab IDs.
  - **(b) `FAPE2DOffsetHead` (`0.00 MB` inference):** Training-only MLP head predicting clamped local-frame relative diatonic pitch-step & octave offsets `(Δstep, Δoct)` between consecutive notes within each bar (stripped in `strip_aux_heads()`).
  - **(c) `DAQRegisterCriticHead` (`~0.1 MB` INT4):** Independent local visual-vs-token consistency critic (`d_daq=96`) trained via BCE on real bars (`y=1`) vs synthetic octave/step-shifted negative bars (`y=0`) to detect silent register shifts.

## 4. Verification & Launch Commands
CPU 2-step smoke test (local, zero GPU):
```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe experiments\2026-11-27T14-55-i15-50-xf-remora-modifiers-daq-fape\run_colab.py --smoke-test
```
Colab A100 / Local 4080 SUPER run:
```bash
python3 -u experiments/2026-11-27T14-55-i15-50-xf-remora-modifiers-daq-fape/run_colab.py
```

## 5. Checkpointing & Remote HF Auto-Resume Plan
- Atomic checkpoint save (`latest.pt.tmp` → `os.replace`) every `5.0` minutes per arm under `WORK_ROOT/ckpts/<arm>/`.
- **1-Arm HF Partial Sync & Startup Auto-Resume:** Checks HF Private Hub on startup for `evals/<EXP_ID>.json` and `checkpoints/<EXP_ID>/{ctrl,remora_daq_fape}_arm_result.json`; uploads Arm 0 immediately upon completion before starting Arm 1.
- Security: reads `HF_TOKEN` from `/tmp/.hf_token` (chmod 600), env `HF_TOKEN`, or `~/.cache/huggingface/token`, never prints it, always unlinks `/tmp/.hf_token` in `finally`.

## 6. Evaluation Gate & Decision Tree
- **Primary Metrics:** `ema_val_ce`, `modifier_val_ce`, `int4_dequant_val_ce`, `daq_register_shift_acc`, `step_ms`, and exact serialized `packed_int4_mb` (`< 55.0 MB` hard budget; `FAPE2DOffsetHead` verified `0.00 MB` at export).
- **PROMOTE:** `remora_daq_fape` beats `ctrl` by `>= -0.015` on `ema_val_ce` (or `>= -0.040` on `modifier_val_ce` with non-regressing overall `ema_val_ce`), `daq_register_shift_acc >= 0.80`, and `packed_int4_mb < 55.0 MB` → port into `uni_omr_v2.py` for Iter 15 from-scratch pretraining.
- **ITERATE:** `modifier_val_ce` or `daq_register_shift_acc` succeeds while overall CE is neutral → ablate `RemoraModifierHead`, `FAPE2DOffsetHead`, and `DAQRegisterCriticHead` individually.
- **DISCARD:** Overall `ema_val_ce` regresses (`> +0.005`) and `modifier_val_ce` gain `< -0.015` → drop cleanly.
