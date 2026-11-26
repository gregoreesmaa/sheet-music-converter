# `2026-11-26T19-35-i13-26b-asymmetric-gqa-cross-mha`

**Title:** Iter 13 Exp 26b — Asymmetric GQA (`n_kv_heads_self=2`, `n_kv_heads_cross=8`) vs Symmetric GQA-4× (`2/2`) and Full MHA (`8/8`) at 800 Steps
**Status:** `QUEUED` (Successor to `2026-11-26T07-40-i13-26-gqa-kv-heads-sweep`)
**Parent Experiment:** [`2026-11-26T07-40-i13-26-gqa-kv-heads-sweep`](../2026-11-26T07-40-i13-26-gqa-kv-heads-sweep/OUTCOME.md)
**Target Platform:** Google Colab `1× NVIDIA L4` (24 GB VRAM, native BF16, ~45–50 min total, ~3.8 CU)

---

## Summary & Hypothesis

In `i13-26`, full MHA (`n_kv_heads=8`) beat the `n_kv_heads=2` control by **`+4.19 pp` EMA Token F1** (`0.2718` vs `0.2299`, EMA SER `0.8626` vs `0.8993`) and beat `n_kv_heads=4` by **`+4.73 pp`**, at only `+1.99 MB` Hybrid INT4+scales (`33.25 MB` vs `31.26 MB`) and identical decode speed (`325.3 tok/s`).

However, two questions remain before locking Iter 14's decoder attention config:
1. **Asymmetric GQA Pareto point (per `i13-26/PLAN.md`):** Can we keep **causal self-attention** at `n_kv_heads_self=2` (`4:1` GQA, `3.75 MB` FP16 self-KV at `L=2048` — `4×` lower autoregressive decode memory bandwidth) while giving **visual cross-attention** full `n_kv_heads_cross=8` (`kv_dim=384`, `36.0 MB` static precomputed cross-KV at `L=3072`)?
2. **Convergence beyond the 400-step EMA/max_len-512 regime:** In `i13-26`, `400` steps with `gpu_ema_decay=0.997` left `30.1%` step-0 weight in EMA and every arm hit `eval_max_len=512`. Here we run **`800` steps/arm** with **`gpu_ema_decay=0.99`** (`0.99^800 = 0.0003`) so both raw and EMA weights reflect genuine convergence.

**3-Arm Controlled Sweep (`800` steps/arm, `seed=42`, `Muon+AdamW`, `lr=3e-4`, `dim=1280`, BF16):**
1. **`kv2_sym`** *(control)*: `n_kv_heads_self=2, n_kv_heads_cross=2` (`15.0 MB` FP16 KV cache)
2. **`asym_s2_c8`** *(treatment)*: `n_kv_heads_self=2, n_kv_heads_cross=8` (`39.75 MB` total; `3.75 MB` self-KV)
3. **`kv8_sym`** *(MHA reference)*: `n_kv_heads_self=8, n_kv_heads_cross=8` (`60.0 MB` FP16 KV cache)

---

## Quick Links
- **Plan:** [PLAN.md](./PLAN.md)
- **Runner:** [run_colab.py](./run_colab.py)
- **HF Artifacts:** `evals/2026-11-26T19-35-i13-26b-asymmetric-gqa-cross-mha.json`, `runs/2026-11-26T19-35-i13-26b-asymmetric-gqa-cross-mha.json`
