# Experiment Plan — `2026-11-26T07-40-i13-09-paretoq-awq-edge-qat`

> **Zero-Prior-Context Handoff:** Everything required to run, monitor, stop, resume, or evaluate `i13-09` (ParetoQ mixed-bit + AWQ channel-scaled edge QAT) is self-contained in this folder (`README.md`, `PLAN.md`, `run_colab.py`).

## 1. Platform & Budget
- **Provider:** Google Colab `1× NVIDIA L4` (`24 GB`, `cc=8.9`, native BF16) via `.\.venv\Scripts\colab.exe`.
- **Session Name:** `i13-09-paretoq-awq`
- **Estimated Wall-Clock & Cost:** `~30 min` (`~2.4 CU`):
  - Shared `UniOMRv2Net(small_45m)` base training (`600` steps on `iter10_5_proxy_2k.tar`): `~7 min`
  - AWQ activation calibration (`32` train pages, forward hooks): `~0.5 min`
  - `5` QAT arms × `200` steps (`lr=1e-4` cosine → `1e-5`, `B=2 × ga=4`) + PTQ & Post-QAT `proxy_val` evals: `~22 min`

## 2. Identifiers
- **Experiment ID:** `2026-11-26T07-40-i13-09-paretoq-awq-edge-qat`
- **HF Repo:** `gregoreesmaa/sheet-music-converter-omr` (`repo_type="dataset"`)
  - Final Eval JSON: `evals/2026-11-26T07-40-i13-09-paretoq-awq-edge-qat.json`
  - Run Ledger JSON: `runs/2026-11-26T07-40-i13-09-paretoq-awq-edge-qat.json`
- **Dataset:** `datasets/iter10_5_proxy_2k.tar` (`2,000` train + `64` val full pages, permissive, 0 `suite31_hw` overlap).

## 3. Exact Launch Commands (PowerShell)
```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\colab.exe new -s i13-09-paretoq-awq --gpu L4
.\.venv\Scripts\colab.exe upload -s i13-09-paretoq-awq <local_bundle.tar.gz> /content/bundle.tar.gz
.\.venv\Scripts\colab.exe upload -s i13-09-paretoq-awq <local_tmp_token> /tmp/.hf_token
.\.venv\Scripts\colab.exe exec -s i13-09-paretoq-awq -f experiments/2026-11-26T07-40-i13-09-paretoq-awq-edge-qat/run_colab.py
.\.venv\Scripts\colab.exe stop -s i13-09-paretoq-awq
```

## 4. Hyperparameters
- Shared base: `UniOMRv2Config.small_45m(grad_checkpoint=True)`, `600` steps (`seed=42`, `lr=3e-4`, `wd=0.04`, `B=2 × ga=4`, `max_dim=1280`, `max_seq_len=1536`).
- All 5 QAT arms: `200` steps (`lr=1e-4` cosine to `1e-5`, `B=2 × ga=4`), block size `32`.
- Metrics per arm (both 0-step PTQ and 200-step QAT): `val_token_f1`, `f1_retention_vs_fp32`, `logit_kl_vs_fp32`, `logit_mse_vs_fp32`, `top1_token_agree`, `val_ce_loss`, `small_45m_webgpu_mb`, and `projected_edge85m_webgpu_mb`.

## 5. Checkpointing & Resume
- Shared base checkpoint saved atomically to `/content/vol/ckpts/<EXP_ID>/base_fp_600.pt`.
- Each arm saves `arm_result.json` on completion and checkpoints every `5.0` minutes.

## 6. Promotion Gate & Decision Tree
- **PROMOTE:**
  - If `int4_awq_scaled_qat` cuts `logit_kl_vs_fp32` by `>= 15%` vs `int4_uniform_qat` with `>= 99.0%` F1 retention → adopt AWQ channel-scaled QAT for Iter 13 `small_45m` & `edge85m`.
  - If `paretoq_w3_ffn_w4_attn` or `paretoq_w2_ffn_w4_attn` achieves `>= 98.5%` F1 retention while cutting WebGPU size by `>= 10%` (`edge85m <= 45 MB`) → adopt ParetoQ mixed-bit QAT to unlock `edge85m` for Iter 13 browser WebGPU deployment!
- **ITERATE:** `paretoq_w2` / `ternary` drops `> 2%` F1 at `200` QAT steps (reflecting ParetoQ's 2-bit representation transition) → test longer QAT (`last 25%` of Iter 13) + KD from FP16 teacher.
- **DISCARD:** Uniform `int4_uniform_qat` dominates all 4 challengers on both F1 retention and KL → keep standard `QATInt4ControllerV2(block_size=32)`.
