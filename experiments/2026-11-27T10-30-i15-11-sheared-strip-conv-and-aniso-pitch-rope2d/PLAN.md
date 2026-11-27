# Plan — `2026-11-27T10-30-i15-11-sheared-strip-conv-and-aniso-pitch-rope2d`

## 1. Platform & Budget
- **Platform:** Google Colab `H100` (`provider: "colab_h100"`, 80 GB VRAM, BF16 native).
- **Estimated Wall-Clock:** ~12 min total for 2 arms (`600` steps/arm on `iter10_5_proxy_2k` + upright and `±6°` tilted-val eval).

## 2. Identifiers
- **Experiment ID:** `2026-11-27T10-30-i15-11-sheared-strip-conv-and-aniso-pitch-rope2d`
- **HF Eval:** `evals/2026-11-27T10-30-i15-11-sheared-strip-conv-and-aniso-pitch-rope2d.json`
- **HF Run:** `runs/2026-11-27T10-30-i15-11-sheared-strip-conv-and-aniso-pitch-rope2d.json`

## 3. Matched Hyperparameters
- **Dataset:** `datasets/iter10_5_proxy_2k.tar` (`1992` train / `64` val + deterministic `±6°` tilted-val stress slice)
- **Backbone/Encoder/Decoder Base:** `UniOMRv2Config.edge85m(n_axial_layers=0, n_global_layers=6, grad_checkpoint=True)`
- **Optimization:** `steps=600`, `seed=42`, `batch_size=2`, `grad_accum=4` (eff BS=8), `muon+adamw`, `lr=3e-4`, `weight_decay=0.01`, `wsd` (`warmup=60`, `decay_frac=0.25`, `min_lr_ratio=0.05`), `bf16`, `gpu_ema_decay=0.997`, `fullpage_max_dim=1280`, `max_seq_len=1536`, `mtp_loss_weight=0.30`, `stroke_hog_weight=0.15`.
- **Arms:**
  1. `ctrl`: stock `AnisotropicMusicBlock` + isotropic `AspectNormalizedRoPE2D`.
  2. `sheared_aniso_rope`: `stage3/stage4` 3-angle gated `(3, 9)` sheared strips (`0°, ±5°`) & `(9, 3)` slanted strips (`0°, ±8°`) + Pitch-Biased Anisotropic 2D-RoPE (`62.5%` Y `base_y=2000, coord_scale_y=256` vs `37.5%` X `base_x=10000, coord_scale_x=128`).
- **Logged Metrics:** `ema_val_ce`, `val_ce`, `ema_val_tilted_ce`, `val_tilted_ce`, `tilt_degradation_ce`, `step_ms`, `peak_vram_mb`, `packed_int4_mb`.
