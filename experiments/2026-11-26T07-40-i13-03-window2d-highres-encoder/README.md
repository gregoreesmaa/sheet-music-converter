# `2026-11-26T07-40-i13-03-window2d-highres-encoder`

**Title:** Iter 13 Academic Frontier Exp 03 — Qwen2.5-VL Shifted 2D Local Window Attention vs 1D Axial Attention at `/8`
**Status:** `PROPOSED` (Iter 13 Academic Batch 01 — OMR & Document VLM Vision-to-Sequence Frontier)
**Baseline Commit SHA:** `9303e51`
**Target Platform:** Google Colab `L4` (24 GB VRAM, BF16 native, ~30–34 min total, ~2.6 CU)

---

## Summary & Hypothesis

`UniOMR v2` runs 2 layers of `AxialEncoderLayer` (1D row attention followed by 1D column attention) on the high-resolution `/8` feature grid before the patch merger. While 1D axial attention is `O(N*(H'+W'))`, it has two structural blind spots for polyphonic and handwritten OMR (`suite31_hw`):
1. **2D Diagonal Neighborhoods Require 2 Hops:** Accidentals to the left of a notehead, diagonal beams, slurs, and ledger lines above a chord cannot interact in a single 1D row or column attention head.
2. **Skewed/Drooping Handwritten Staves:** Under phone-photo rotation (`±10°`) or wavy historical manuscript staff lines, a 1D horizontal row slices *diagonally across* neighboring staves rather than following the staff!

Inspired by **Qwen2.5-VL**'s native 2D **Window Attention** (*Bai et al., arXiv:2502.13923*, Feb 2025), **DeepSeek-OCR**'s SAM-window + global **DeepEncoder**, and **TrOMR** (*Li et al., ICASSP 2024 / arXiv:2308.09370*), we replace or pair the `/8` `AxialEncoderLayer` stack with **Shifted 2D Local Window Attention with 2D-RoPE** (which actually uses **fewer parameters** and **lower FLOPs** than 2-pass Row+Col Axial Attention!):
1. `axial_control`: 2× `AxialEncoderLayer` (exact current `UniOMR v2` baseline).
2. `win2d_8x8_shifted`: 2× shifted `8×8` (`64px × 64px`) 2D-window attention with 2D-RoPE (`-1.18M` params vs axial!).
3. `win2d_8x16_staff_strip`: 2× shifted `8×16` (`64px × 128px` horizontal staff-strip) 2D-window attention with 2D-RoPE (`-1.18M` params vs axial!).
4. `hybrid_win8x16_then_axial`: Layer 0 = `8×16` 2D-window attention (local 2D chord/accidental/skew fusion) -> Layer 1 = `AxialEncoderLayer` (full-width staff-line & full-height system context; `-0.59M` params vs axial!).

**Hypothesis:** `win2d_8x16_staff_strip` or `hybrid_win8x16_then_axial` will capture 2D glyph neighborhoods and tilted/drooping staves cleanly at `/8` resolution, improving held-out full-page `proxy_val` Token F1 by `>= +0.5 pp` while simultaneously **reducing** parameter count (`-0.59M` to `-1.18M`) and increasing L4/WebGPU throughput.

---

## Quick Links

- **Plan & Runbook:** [PLAN.md](./PLAN.md)
- **Standalone Colab Runner:** [run_colab.py](./run_colab.py)
- **Outcome:** `OUTCOME.md` *(written after Colab run completes)*
- **HF Private Hub Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-26T07-40-i13-03-window2d-highres-encoder.json`
  - Run Ledger: `runs/2026-11-26T07-40-i13-03-window2d-highres-encoder.json`
