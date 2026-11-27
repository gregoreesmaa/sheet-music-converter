# PLAN — `2026-11-27T14-15-i15-33-diff-cross-attn`

## 1. Platform & Budget
- **Platform:** Google Colab `1× NVIDIA A100` (or Local `RTX 4080 SUPER` 16 GB via `D:\tmp\run_local_p0_queue.py`, native BF16).
- **Estimated Wall-Clock:** ~25 min total on A100 (2 arms × 600 steps + val CE + exact Hybrid INT4 packing & dequant eval).
- **Baseline Commit SHA:** `a32c810`.
- **Self-Containment:** All experimental `DiffCrossAttention` code lives strictly inside `run_colab.py`; `tmp-train/mvd/uni_omr_v2.py` is untouched.

## 2. Identifiers for Follow-Up Queries
- **Experiment ID:** `2026-11-27T14-15-i15-33-diff-cross-attn`
- **Hugging Face (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-27T14-15-i15-33-diff-cross-attn.json`
  - Run Ledger: `runs/2026-11-27T14-15-i15-33-diff-cross-attn.json`
  - Partial Arm Sync: `checkpoints/2026-11-27T14-15-i15-33-diff-cross-attn/{ctrl,diff_ca}_arm_result.json`

## 3. Data & Controlled 2-Arm Protocol
- **Dataset:** `datasets/iter10_5_proxy_2k.tar` (`2,000` train / `64` val full pages, permissive licenses only, `0` `CC-BY-NC`, `2×` dense multi-staff & HW sampling weight, `FULLPAGE_MAX_DIM=1280`, `MAX_SEQ_LEN=1536`).
- **Shared Hyperparameters:** `600` steps/arm, `seed=42`, BF16 autocast, `batch_size=2`, `grad_accum=4` (eff. batch 8), **verified `muon+adamw`** (`lr=3e-4`, `weight_decay=0.04`, `no_decay_1d=True`; asserts `opt.mode == "muon+adamw"` on CUDA so no silent AdamW fallback is possible), WSD cosine schedule (`warmup=60`, `decay_frac=0.25`, `min_lr_ratio=0.05`), EMA (`decay=0.99` every step), `QATInt4ControllerV2` enabled at step `420` (final 30%), `grad_checkpoint=True`.
- **Windows RTX 4080 SUPER Safety:** Automatically disables `torch.backends.cudnn.enabled` and forces CUTLASS `EFFICIENT_ATTENTION` on `win32`.
- **Arm 0 (`ctrl`):** `UniOMRv2Config.edge85m(n_axial_layers=0, n_global_layers=6, grad_checkpoint=True)` with stock GQA cross-attention (`d_qk=64, d_v=64`).
- **Arm 1 (`diff_ca`):** Same `edge85m + ax0_gl6` config, replacing all 10 decoder `cross_attn` layers with `DiffCrossAttention` (Ye et al. 2024): splits Q and K into two `32`-dim halves `(Q1, K1)` and `(Q2, K2)` with RMSNorm per half, learnable per-layer `lambda = exp(lam_q1·lam_k1) - exp(lam_q2·lam_k2) + lambda_init(l)` where `lambda_init(l) = 0.8 - 0.6 * exp(-0.3 * l)`, computes `RMSNorm(Attn(Q1, K1, V) - lambda * Attn(Q2, K2, V)) * (1 - lambda_init)`. Packs `(K1, K2)` back into `(B, n_kv_heads, Lk, 64)` so KV-cache RAM is unchanged (`+0.004 MB` INT4 weights).

## 4. Verification & Launch Commands
CPU 2-step smoke test (local, zero GPU):
```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe experiments\2026-11-27T14-15-i15-33-diff-cross-attn\run_colab.py --smoke-test
```
Colab A100 / Local 4080 SUPER run:
```bash
python3 -u experiments/2026-11-27T14-15-i15-33-diff-cross-attn/run_colab.py
```

## 5. Checkpointing & Remote HF Auto-Resume Plan
- Atomic checkpoint save (`latest.pt.tmp` → `os.replace`) every `5.0` minutes per arm under `WORK_ROOT/ckpts/<arm>/` with full step/optimizer/EMA/RNG state for mid-arm local resume.
- **1-Arm HF Partial Sync & Startup Auto-Resume:** On startup, checks HF Private Hub for `evals/<EXP_ID>.json` and `checkpoints/<EXP_ID>/{ctrl,diff_ca}_arm_result.json` and skips any already-completed arm. After Arm 0 (`ctrl`) finishes, immediately uploads both `checkpoints/<EXP_ID>/ctrl_arm_result.json` and partial `evals/<EXP_ID>.json` (`status: "RUNNING"`) to HF before starting Arm 1 (`diff_ca`).
- Security: reads `HF_TOKEN` from `/tmp/.hf_token` (chmod 600), env `HF_TOKEN`, or `~/.cache/huggingface/token`, never prints it, always unlinks `/tmp/.hf_token` in `finally`.

## 6. Evaluation Gate & Decision Tree
- **Primary Metrics:** `ema_val_ce`, `int4_dequant_val_ce`, `step_ms`, and exact serialized `packed_int4_mb` (`< 55.0 MB` hard budget).
- **PROMOTE:** `diff_ca` beats `ctrl` by `>= -0.015` on `ema_val_ce` (and `int4_dequant_val_ce`), stays `< 55.0 MB` packed INT4, and `step_ms` overhead is `<= +10%` → port `DiffCrossAttention` into `uni_omr_v2.py` for Iter 15 from-scratch pretraining.
- **ITERATE:** Beats `ctrl` by `-0.005` to `-0.015` or shows strong gain on degraded/noisy pages with slight speed overhead → test applying `DiffCrossAttention` only to the first 4 decoder layers (or combining with `CoordKey-CA`).
- **DISCARD:** `< -0.005` gain on `ema_val_ce` over `ctrl` at 600 steps (or unstable `lambda` / INT4 degradation `> +0.025`) → keep stock GQA cross-attention.
