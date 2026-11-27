# `2026-11-27T10-30-i15-38-coordkey-ca-fourier-cross-attn` — Decoupled 2D Fourier `CoordKey-CA`

- **Experiment ID:** `2026-11-27T10-30-i15-38-coordkey-ca-fourier-cross-attn`
- **Short ID / Wave:** `i15-38` (Iter 15 — Weight-Incompatible Architecture Proxy)
- **Status:** `PLANNED`
- **Baseline Commit SHA:** `cfc94aa`
- **Platform / GPU:** Colab `1× NVIDIA H100` (`80 GB` VRAM, BF16 native, ~12 min total for 2×600 steps)
- **Baseline Architecture:** `UniOMRv2Config.edge85m(n_axial_layers=0, n_global_layers=6)` (`d_model=512`, 6 global 2D-RoPE encoder layers, 10 GQA decoder layers with `d_qk=64, d_v=64`, ~80M infer params, `< 55 MB` Hybrid FP16/INT4)
- **Dataset (100% permissive):** `gregoreesmaa/sheet-music-converter-omr` → `datasets/iter10_5_proxy_2k.tar` (`2,000` train / `64` val full pages, `2×` dense multi-staff & HW sampling weight)

## Summary & Hypothesis
In stock `UniOMRv2Net`, decoder cross-attention has no explicit 2D positional mechanism beyond an additive `pos2d` vector mixed into `mem_final`. Under `edge85m` (`d_qk=64, n_heads=8, n_kv_heads=2`), content matching (glyph identity) and 2D spatial cursor addressing (system/measure/voice tracking) must compete for the same 64-dimensional QK dot-product subspace — contributing to `<sys>` vertical skips and measure-tracking drift on dense polyphonic or handwritten pages.

This controlled 2-arm weight-incompatible proxy (`coordkey_ca` vs `ctrl`) introduces a **shared 2D Fourier `CoordMLP (66 -> 128 -> 128)`** (2 normalized `(y, x)` coords + 16 log-spaced `(sin, cos)` frequencies per axis) and concatenates a **dedicated 16-dim RMSNorm'd spatial subspace** onto cross-attention Q & K (`d_qk = 64 + 16 = 80, d_v = 64`) in all 10 decoder layers. Value vectors and output projections remain untouched (`d_v = 64`), keeping packed INT4 footprint overhead negligible (`~+0.47 MB`, well under `< 55 MB`) while decoupling spatial cursor addressing from visual glyph matching.

## Quick Links
- Execution Plan & Decision Tree: [PLAN.md](./PLAN.md)
- Self-Contained Colab H100 Runner: [run_colab.py](./run_colab.py)
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-27T10-30-i15-38-coordkey-ca-fourier-cross-attn.json`
- HF Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-27T10-30-i15-38-coordkey-ca-fourier-cross-attn.json`
