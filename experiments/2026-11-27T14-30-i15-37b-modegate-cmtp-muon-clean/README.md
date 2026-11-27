# `2026-11-27T14-30-i15-37b-modegate-cmtp-muon-clean`

## Context & Hypothesis

Clean apples-to-apples re-verification of [`2026-11-27T10-30-i15-37-modegate-dualhead-text-and-chained-mtp`](../2026-11-27T10-30-i15-37-modegate-dualhead-text-and-chained-mtp/OUTCOME.md), where Arm 1 (`modegate_cmtp`) posted strong gains (`ema_val_ce -0.10038`, `val_ce -0.04395`, `spec_accept_len +0.2481` EMA / `+0.1840` raw) but was **confounded** because the `"base."` wrapper prefix caused `partition_muon_adamw_params()` to fall back to `opt=adamw` on Arm 1 while `ctrl` ran `opt=muon+adamw`.

With the `"base."` prefix stripping fix landed in `tmp-train/mvd/train_uni_v2.py` (`95446af` / `b64fd64`), `i15-37b` enforces `assert opt.mode == "muon+adamw"` on **both** arms and upgrades the protocol to match `i15-38` / `i15-33` / `i15-14`:
- `EMA_DECAY = 0.99` (removing the 333-step lag artifact on 600 steps),
- `QATInt4ControllerV2` at step `420` + exact disk-verified Hybrid FP16/INT4 `int4_dequant_val_ce`,
- HF Private Hub per-arm partial checkpoint auto-resume.

## Decision Gate

- **PROMOTE `modegate_cmtp`** if (with `opt.mode == "muon+adamw"` verified on both arms):
  - `ema_val_ce` matches or beats `ctrl` (`<= ctrl + 0.010` nats, ideally `< ctrl`) AND `int4_dequant_val_ce <= ctrl + 0.010` (quantization delta `<= +0.025`), AND
  - Greedy-chained `spec_accept_len` improves by `>= +0.10` tokens/step (or `mtp_k2..k4_acc >= +3.0 pp`), AND
  - `packed_int4_with_mtp_mb <= 55.0 MB` and `step_ms <= +10%` vs `ctrl`.
- **ITERATE** if Chained MTP-4 improves `spec_accept_len` cleanly (`>= +0.10`) but Dual-Head / `text_aux_loss` regresses `ema_val_ce` or `int4_dequant_val_ce` (isolate Chained-MTP alone).
- **DISCARD** if `modegate_cmtp` regresses `ema_val_ce` by `> 0.025` nats or fails to gain `>= +0.08` `spec_accept_len` under matched `muon+adamw`.
