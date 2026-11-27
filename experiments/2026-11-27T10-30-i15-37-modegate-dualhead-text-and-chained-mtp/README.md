# `2026-11-27T10-30-i15-37-modegate-dualhead-text-and-chained-mtp`

## Hypothesis

Stock `UniOMRv2Net` (`edge85m + ax0_gl6`) uses:
1. **Independent parallel `MTP-4` heads** (`Linear(D, D)` on `h_0` alone), which cannot condition depth `k` on the predicted token `y_{t+k-1}` — causing speculative draft accuracy to collapse steeply at `k=3, 4` on rhythmic/pitch chains.
2. **A single shared `lm_head`** with modality-agnostic decoder cross-attention, which forces future inline text/lyric/directive tokens to compete directly with music notation queries and output rows.

This 2-arm weight-incompatible proxy tests both upgrades jointly on `edge85m + ax0_gl6` (`600` steps/arm, `iter10_5_proxy_2k`, BF16, Muon+AdamW, seed `42`):
- **Arm 0 (`ctrl`)**: Stock `edge85m + ax0_gl6` with parallel `MTP-4` (`Linear(D, D)`) + single `lm_head`.
- **Arm 1 (`modegate_cmtp`)**:
  - Replaces independent parallel `MTPHead`s with DeepSeek-V3 sequential **Chained MTP-4**: `h_k = h_{k-1} + SiLU(W_k([RMSNorm(h_{k-1}); RMSNorm(tok_emb(y_{t+k}))]))`.
  - Adds a lightweight **Modality-Gated Dual-Head**: `lm_head_text` (129-tok ASCII/byte + `<text-end>`) + layers `0..3` `cross_attn` FiLM & rank-32 text-query delta (zero-initialized for identity start).

## Decision Gate

- **PROMOTE `modegate_cmtp`** if:
  - `ema_val_ce` matches or beats `ctrl` (`<= ctrl + 0.015` nats, ideally `< ctrl`), AND
  - Greedy-chained speculative acceptance proxy (`mtp_k2..k4_acc` / `spec_accept_len`) improves by `>= +3.0 pp` at `k>=2` (or `>= +0.10` expected accepted tokens/step), AND
  - `packed_int4_with_mtp_mb <= 55.0 MB` (preserving WebGPU budget).
- **ITERATE** if Chained MTP-4 improves speculative acceptance cleanly but Dual-Head overhead or auxiliary text loss perturbs `ema_val_ce` (split into isolated Chained-MTP vs ModeGate ablations).
- **DISCARD** if `modegate_cmtp` regresses `ema_val_ce` by `> 0.03` nats or adds `> 10%` `step_ms` without speculative acceptance gain.
