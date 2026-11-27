# Plan — `2026-11-27T10-30-i15-37-modegate-dualhead-text-and-chained-mtp`

## 1. Platform & Budget
- **Platform:** Google Colab `H100` (`provider: "colab_h100"`, 80 GB VRAM, BF16 native).
- **Estimated Wall-Clock:** ~13 min total for 2 arms (`600` steps/arm on `iter10_5_proxy_2k`).

## 2. Identifiers
- **Experiment ID:** `2026-11-27T10-30-i15-37-modegate-dualhead-text-and-chained-mtp`
- **HF Eval:** `evals/2026-11-27T10-30-i15-37-modegate-dualhead-text-and-chained-mtp.json`
- **HF Run:** `runs/2026-11-27T10-30-i15-37-modegate-dualhead-text-and-chained-mtp.json`

## 3. Matched Hyperparameters
- **Dataset:** `datasets/iter10_5_proxy_2k.tar` (`1992` train / `64` val)
- **Backbone/Encoder/Decoder Base:** `UniOMRv2Config.edge85m(n_axial_layers=0, n_global_layers=6, grad_checkpoint=True)`
- **Optimization:** `steps=600`, `seed=42`, `batch_size=2`, `grad_accum=4` (eff BS=8), `muon+adamw`, `lr=3e-4`, `weight_decay=0.01`, `wsd` (`warmup=60`, `decay_frac=0.25`, `min_lr_ratio=0.05`), `bf16`, `gpu_ema_decay=0.997`, `fullpage_max_dim=1280`, `max_seq_len=1536`, `mtp_loss_weight=0.30`, `stroke_hog_weight=0.15`.
- **Arms:**
  1. `ctrl`: stock parallel `MTP-4` + single `lm_head`.
  2. `modegate_cmtp`: DeepSeek-V3 sequential Chained `MTP-4` (`W_k([RMSNorm(h_{k-1}); RMSNorm(tok_emb(y_{t+k}))])`) + Modality-Gated Dual-Head (`lm_head_text` 129-tok + layers `0..3` `cross_attn` FiLM + rank-32 text-query delta).
- **Logged Metrics:** `ema_val_ce`, `val_ce`, `mtp_k1_acc..mtp_k4_acc` (greedy-chained speculative acceptance proxy + teacher-forced diagnostic), `spec_accept_len`, `step_ms`, `peak_vram_mb`, `packed_int4_stripped_mb`, `packed_int4_with_mtp_mb`.
