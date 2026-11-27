# `2026-11-27T10-30-i15-36-yoco-cla-resformer-12l-decoder` — YOCO-3Bank Cross-KV + CLA-2:1 + ResFormer 12L Decoder Proxy

- **Experiment ID:** `2026-11-27T10-30-i15-36-yoco-cla-resformer-12l-decoder`
- **Short ID / Wave:** `i15-36` (Wave 13 — Weight-Incompatible Architecture Proxies)
- **Status:** `QUEUED`
- **Baseline Commit SHA:** `cfc94aa`
- **Platform / GPU:** Colab `1× NVIDIA H100` (`80 GB` VRAM, BF16 native, ~10–12 min total)
- **Arms (`600` steps/arm, BF16, `seed=42`, `grad_accum=4`, WSD, EMA `0.99`, QAT-INT4 at `step >= 420`):**
  - **Arm 0 (`ctrl`):** `edge85m + ax0_gl6` (`d_model=512`, 10L decoder, `d_ff=2048`, GQA-2 independent self-KV & cross-KV).
  - **Arm 1 (`yoco12`):** `edge85m + ax0_gl6` with 12L decoder (`d_ff=1664`), **YOCO-3Bank Cross-KV** (`n_kv=4`, 3 banks shared by layers `0..2`, `3..6`, `7..11`), **CLA-2:1 Self-KV** (`n_kv=2`, 6 banks shared by pairs `(0,1)..(10,11)`), and **ResFormer `v_res`** (`v_l = (1 - sigmoid(a)) * v_l + sigmoid(a) * v_0`).
- **Dataset (100% permissive):** `gregoreesmaa/sheet-music-converter-omr` → `datasets/iter10_5_proxy_2k.tar` (`2,000` train + `64` val full pages, `2×` dense multi-staff & HW upsampling).

## Summary & Hypothesis
In `edge85m`, 10 independent GQA-2 decoder layers materialize 10 separate self-KV caches and 10 separate cross-KV caches (`~26.2 MB` FP16 at `L_mem=3072, T=2048`), creating both WebGPU memory-bandwidth pressure and a bottleneck in cross-attention visual head width (`n_kv=2`).

By combining **YOCO-3Bank Cross-KV** (3 cross-KV banks at `n_kv=4` aligned with DeepStack levels across layers `0..2`, `3..6`, `7..11`), **CLA-2:1 Self-KV** (6 self-KV banks at `n_kv=2`), and **ResFormer value-residuals** (`v_l = (1 - σ(a)) * v_l + σ(a) * v_0`), we can deepen the decoder from `10L → 12L` at iso-params (`d_ff=1664`, `< 55 MB` packed Hybrid INT4) while cutting total KV-cache footprint by **40%** (`~26.2 MB → ~15.7 MB`) and doubling cross-attention KV head width.

## Quick Links
- Execution Plan & Decision Tree: [PLAN.md](./PLAN.md)
- Standalone Colab H100 Runner: [run_colab.py](./run_colab.py)
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-27T10-30-i15-36-yoco-cla-resformer-12l-decoder.json`
- HF Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-27T10-30-i15-36-yoco-cla-resformer-12l-decoder.json`
