# PLAN — `2026-11-27T10-30-i15-38-coordkey-ca-fourier-cross-attn`

## 1. Platform & Budget
- **Platform:** Google Colab `1× NVIDIA H100` (`80 GB` VRAM, native BF16).
- **Estimated Wall-Clock:** ~12 min total (2 arms × 600 steps + val CE + exact Hybrid INT4 packing & dequant eval).
- **Baseline Commit SHA:** `cfc94aa`.
- **Self-Containment:** All experimental `FourierCoordMLP` and `CoordKeyCrossAttention` modules live strictly inside `run_colab.py`; `tmp-train/mvd/uni_omr_v2.py` is untouched.

## 2. Identifiers for Follow-Up Queries
- **Experiment ID:** `2026-11-27T10-30-i15-38-coordkey-ca-fourier-cross-attn`
- **Hugging Face (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-27T10-30-i15-38-coordkey-ca-fourier-cross-attn.json`
  - Run Ledger: `runs/2026-11-27T10-30-i15-38-coordkey-ca-fourier-cross-attn.json`

## 3. Data & Controlled 2-Arm Protocol
- **Dataset:** `datasets/iter10_5_proxy_2k.tar` (`2,000` train / `64` val full pages, `2×` dense multi-staff & HW sampling weight, `FULLPAGE_MAX_DIM=1280`, `MAX_SEQ_LEN=1536`).
- **Shared Hyperparameters:** `600` steps/arm, `seed=42`, BF16 autocast, `batch_size=2`, `grad_accum=4` (eff. batch 8), Muon + AdamW (`lr=3e-4`, `weight_decay=0.04`, `no_decay_1d=True`), WSD cosine schedule (`warmup=60`, `decay_frac=0.25`, `min_lr_ratio=0.05`), EMA (`decay=0.99` every step), `QATInt4ControllerV2` enabled at step `420` (final 30%), `grad_checkpoint=True`.
- **Arm 0 (`ctrl`):** `UniOMRv2Config.edge85m(n_axial_layers=0, n_global_layers=6, grad_checkpoint=True)` with stock cross-attention (`d_qk=64, d_v=64`, additive `pos2d`).
- **Arm 1 (`coordkey_ca`):** Same `edge85m + ax0_gl6` config, adding shared 2D Fourier `CoordMLP (66 -> 128 -> 128)` (from router-indexed `(y_norm, x_norm)` coordinates) + dedicated 16-dim RMSNorm'd spatial subspace concatenated onto Q & K (`d_qk = 64 + 16 = 80, d_v = 64`) in all 10 decoder cross-attention layers. Supports both training and KV-cached inference (`precompute_cross_kv`).

## 4. Verification & Launch Commands
CPU smoke test (local, zero GPU):
```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe experiments\2026-11-27T10-30-i15-38-coordkey-ca-fourier-cross-attn\run_colab.py --smoke-test
```
Colab H100 run:
```bash
python3 -u experiments/2026-11-27T10-30-i15-38-coordkey-ca-fourier-cross-attn/run_colab.py
```

## 5. Checkpointing & Resume Plan
- Atomic checkpoint save (`latest.pt.tmp` → `os.replace`) every `5.0` minutes per arm under `/content/i15_38/ckpts/<arm>/`.
- Per-arm `arm_result.json` cache skips finished arms on resume.
- Security: reads `HF_TOKEN` from `/tmp/.hf_token` (chmod 600) or env, never prints it, always unlinks `/tmp/.hf_token` in `finally`.

## 6. Evaluation Gate & Decision Tree
- **Primary Metrics:** `ema_val_ce`, `int4_dequant_val_ce`, `step_ms`, and exact serialized `packed_int4_mb` (`< 55.0 MB` hard budget).
- **PROMOTE:** `coordkey_ca` beats `ctrl` by `>= -0.015` on `ema_val_ce` (and `int4_dequant_val_ce`), stays `< 55.0 MB` packed INT4, and `step_ms` overhead is `<= +10%` → port `FourierCoordMLP` + `CoordKeyCrossAttention` into `uni_omr_v2.py` for next from-scratch pretraining run.
- **ITERATE:** Improves `ema_val_ce` by `>= -0.01` but INT4 quantization degrades `q_coord_proj`/`k_coord_proj` (`int4_dequant_val_ce - ema_val_ce > 0.025`) → keep tiny spatial projections in FP16 (`~+0.5 MB`).
- **DISCARD:** `< -0.005` gain on `ema_val_ce` over `ctrl` at 600 steps → keep stock `d_qk=64` cross-attention.
