# `2026-11-27T10-30-i15-31-hrnet-music-dualres-s4` — HRNet-Music Dual-Resolution `stage4` (`/4 || /8`)

- **Experiment ID:** `2026-11-27T10-30-i15-31-hrnet-music-dualres-s4`
- **Short ID / Wave:** `i15-31` (Iter 15 — Weight-Incompatible Architecture Proxy)
- **Status:** `PLANNED`
- **Baseline Commit SHA:** `cfc94aa`
- **Platform / GPU:** Colab `1× NVIDIA H100` (`80 GB` VRAM, BF16 native, ~12–14 min total for 2×600 steps)
- **Baseline Architecture:** `UniOMRv2Config.edge85m(n_axial_layers=0, n_global_layers=6)` (`d_model=512`, 6 global 2D-RoPE encoder layers, 10 GQA decoder layers, ~80M infer params, `< 55 MB` Hybrid FP16/INT4)
- **Dataset (100% permissive):** `gregoreesmaa/sheet-music-converter-omr` → `datasets/iter10_5_proxy_2k.tar` (`2,000` train / `64` val full pages, `2×` dense multi-staff & HW sampling weight)

## Summary & Hypothesis
In stock `AnisotropicMusicBackbone`, `stage4` runs 3 serial `AnisotropicMusicBlock(448)` blocks purely at `/8` resolution, with `P2 (/4)` only fused once via a static `PixelUnshuffle(2)` lateral projection (`p2_to_p4`). Because `backbone` weights remain FP16 in Hybrid FP16/INT4 export, serial `stage4 (448ch)` + `p2_to_p4` + `proj (448->512)` costs `~3.85M` FP16 params (`~7.70 MB` of our `< 55 MB` WebGPU budget) while still blurring fine `/4` features (accidentals, augmentation dots, flags, ledger lines) through 3 consecutive `/8` blocks.

This controlled 2-arm weight-incompatible proxy (`hrnet_music_s4` vs `ctrl`) replaces serial `stage4 (448ch)` + `p2_to_p4` + `proj` with a **parallel dual-resolution `/4 (96ch) || /8 (320ch)` bidirectional-exchange `stage4`** (3 parallel anisotropic blocks with `PixelUnshuffle(2)` `/4 -> /8` and bilinear `/8 -> /4` cross-resolution exchange at every block, followed by a final `SpaceToDepth` concat `320 + 4*96 = 704 -> 512` projection). We hypothesize this simultaneously **saves `~1.94 MB`** of FP16 backbone budget AND improves fine-glyph `ema_val_ce` & `int4_dequant_val_ce`.

## Quick Links
- Execution Plan & Decision Tree: [PLAN.md](./PLAN.md)
- Self-Contained Colab H100 Runner: [run_colab.py](./run_colab.py)
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-27T10-30-i15-31-hrnet-music-dualres-s4.json`
- HF Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-27T10-30-i15-31-hrnet-music-dualres-s4.json`
