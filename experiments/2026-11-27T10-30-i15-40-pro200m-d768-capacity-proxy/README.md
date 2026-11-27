# `2026-11-27T10-30-i15-40-pro200m-d768-capacity-proxy` — `pro200m` (`d=768`, `~120 MB` INT4) Capacity Tier Proxy

- **Experiment ID:** `2026-11-27T10-30-i15-40-pro200m-d768-capacity-proxy`
- **Short ID / Wave:** `i15-40` (Wave 13 — Weight-Incompatible Architecture Proxies)
- **Status:** `QUEUED`
- **Baseline Commit SHA:** `cfc94aa`
- **Platform / GPU:** Colab `1× NVIDIA H100` (`80 GB` VRAM, BF16 native, ~14 min total)
- **Arms (`600` steps/arm, BF16, `seed=42`, `grad_accum=4`, WSD, EMA `0.99`, QAT-INT4 at `step >= 420`):**
  - **Arm 0 (`edge85m_ctrl`):** `edge85m + ax0_gl6` (`d_model=512`, `n_heads=8`, `n_kv_heads=2`, `n_axial_layers=0`, `n_global_layers=6`, `n_dec_layers=10`, `ffn_mult=4.0` → `~78M` infer params, `~53 MB` Hybrid INT4).
  - **Arm 1 (`pro200m`):** `pro200m + ax0_gl6` (`d_model=768`, `n_heads=12`, `n_kv_heads=4`, `n_axial_layers=0`, `n_global_layers=8`, `n_dec_layers=12`, `ffn_mult=3.5` → `~185–205M` infer params, `~115–128 MB` Hybrid INT4 — shippable on all last-3-gen smartphones & laptops + elite KD teacher for `edge85m`!).
- **Dataset (100% permissive):** `gregoreesmaa/sheet-music-converter-omr` → `datasets/iter10_5_proxy_2k.tar` (`2,000` train + `64` val full pages, `2×` dense multi-staff & HW upsampling).

## Summary & Hypothesis
Scaling from `small_45m` (`d=384`) to `edge85m` (`d=512`) unlocked major dense-polyphony and handwritten OMR gains in `i14-05`. While `edge85m` (`~53 MB` INT4) remains our ultra-light WebGPU default tier, modern laptops and last-3-generation smartphones comfortably host a `~120 MB` INT4 **Pro Tier** (`pro200m`, `d=768`, 8 global encoder layers, 12 decoder layers), which can also serve as a high-capacity Knowledge Distillation teacher for `edge85m`.

This controlled 2-arm proxy measures the exact scaling payoff of `pro200m` over `edge85m_ctrl` at 600 steps on `iter10_5_proxy_2k`, tracking `ema_val_ce`, `int4_dequant_val_ce`, `step_ms`, `vram_gb`, and exact on-disk `packed_int4_mb`.

## Quick Links
- Execution Plan & Decision Tree: [PLAN.md](./PLAN.md)
- Standalone Colab H100 Runner: [run_colab.py](./run_colab.py)
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-27T10-30-i15-40-pro200m-d768-capacity-proxy.json`
- HF Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-27T10-30-i15-40-pro200m-d768-capacity-proxy.json`
