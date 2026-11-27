# `2026-11-27T14-15-i15-33-diff-cross-attn`

- **Status:** `PLANNED` (QUEUED in `experiments/overnight_queue/queue.json`, Wave 1)
- **Short ID:** `i15-33`
- **Baseline Commit SHA:** `a32c810`

## Summary & Hypothesis
Tests **Differential Cross-Attention (`DiffCrossAttn`, Ye et al. 2024)** in all 10 decoder cross-attention layers of `UniOMRv2Net(edge85m, ax0_gl6, d=512)` against the stock GQA cross-attention control (`ctrl`) under a controlled 2-arm 600-step BF16 Muon+AdamW proxy on `iter10_5_proxy_2k.tar`.

Each cross-attention layer splits its Q and K head dimensions (`head_dim=64 -> two 32-dim halves`) into `(Q1, K1)` and `(Q2, K2)` with a learnable per-layer scalar `lambda = exp(lam_q1·lam_k1) - exp(lam_q2·lam_k2) + lambda_init(l)` and computes `RMSNorm(Attn(Q1, K1, V) - lambda * Attn(Q2, K2, V)) * (1 - lambda_init)`. Subtracting a second cross-attention map suppresses common-mode visual noise (blank-staff lines, bleed-through, foxing, desk shadows) at `+0.004 MB` INT4 weight cost and `0%` extra KV-cache RAM.

## Quick Links
- [PLAN.md](PLAN.md)
- [run_colab.py](run_colab.py)
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-27T14-15-i15-33-diff-cross-attn.json`
- HF Partial Sync: `checkpoints/2026-11-27T14-15-i15-33-diff-cross-attn/`
