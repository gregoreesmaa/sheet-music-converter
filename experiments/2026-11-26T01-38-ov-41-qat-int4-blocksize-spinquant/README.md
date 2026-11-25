# `2026-11-26T01-38-ov-41-qat-int4-blocksize-spinquant` — Overnight Exp 41: QAT-INT4 `block_size` × SpinQuant Hadamard Sweep

- **Status:** `PLANNED` (`QUEUED` in `experiments/overnight_queue/batch_10.json`, Wave 8)
- **Baseline Commit SHA:** `164852f`
- **Model:** `UniOMR v2 small_45m` (`~43M` total params) with `QATInt4ControllerV2` & `spinquant_fold.fold_hadamard_v2`
- **Dataset (100% permissive):**
  - `gregoreesmaa/sheet-music-converter-omr` → `datasets/iter10_5_proxy_2k.tar` (`2,000` train + `64` val full pages)

## Summary & Hypothesis
Symmetric group-wise INT4 quantization (`MatMulNBits`) for browser WebGPU deployment trades off group-scale overhead (`2 bytes / block_size` per weight) against quantization fidelity (`block_size=32` vs `64` vs `128`). Offline **SpinQuant** Hadamard rotation folding (`fold_hadamard_v2`) suppresses channel-wise activation and weight outliers with **zero runtime ops**, which should dramatically reduce INT4 quantization error—especially at coarser `block_size=64` and `128`.

To eliminate initialization and pretraining variance, we train a single shared `small_45m` BF16 base model for `600` steps on `iter10_5_proxy_2k.tar`, record its FP32/BF16 reference `proxy_val` metrics and logits, and branch into a controlled **3 × 2 factorial matrix** (`block_size ∈ {32, 64, 128}` × `SpinQuant ∈ {Off, On}`), measuring BOTH **0-step PTQ** and **200-step QAT-INT4** post-quantization `val_token_f1`, `f1_retention_vs_fp32`, `logit_kl_vs_fp32`, `logit_mse_vs_fp32`, and exact WebGPU model size (`MB`).

**Hypothesis:** Combining `fold_hadamard_v2` with QAT-INT4 will reduce post-INT4 logit KL by `>= 20%` and enable `block_size=64` (saving `~1.1 MB` of FP16 scale overhead vs `block_size=32`) to match or exceed unrotated `block_size=32` F1 retention.

## Quick Links
- Colab Launcher: [run_colab.py](./run_colab.py)
- Plan & Decision Tree: [PLAN.md](./PLAN.md)
- Queue Entry: [batch_10.json](../overnight_queue/batch_10.json)
- SpinQuant Implementation: [spinquant_fold.py](../../tmp-train/mvd/spinquant_fold.py)
- Outcome: `OUTCOME.md` *(written immediately upon completion)*
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-26T01-38-ov-41-qat-int4-blocksize-spinquant.json`
- HF Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-26T01-38-ov-41-qat-int4-blocksize-spinquant.json`
