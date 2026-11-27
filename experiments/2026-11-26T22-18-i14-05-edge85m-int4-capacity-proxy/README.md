# `2026-11-26T22-18-i14-05-edge85m-int4-capacity-proxy` — `edge85m` `< 55 MB` INT4 Capacity Scaling Proxy (Medium)

- **Experiment ID:** `2026-11-26T22-18-i14-05-edge85m-int4-capacity-proxy`
- **Short ID / Wave:** `i14-05` (Wave 11 — Optical Resolution & Capacity Wall)
- **Status:** `COMPLETED (PROMOTE)` — see [OUTCOME.md](./OUTCOME.md)
- **Baseline Commit SHA:** `87b1ec8`
- **Platform / GPU:** Modal `1× NVIDIA L4` (`24 GB` VRAM, BF16 native, ~45 min, ~3.6 CU; Colab L4 compatible)
- **Model Footprint:**
  - **Arm A (`small_45m`):** `d_model=384`, 8 dec layers, `~42M` infer params (`~28–31 MB` Hybrid FP16/INT4)
  - **Arm B (`edge85m`):** `d_model=512`, 10 dec layers, `~80M` infer params (`~50–53 MB` Hybrid FP16/INT4 — verified on-disk `< 55.0 MB` after real 4-bit nibble packing + FP16 group-32 scales!)
- **Datasets (100% permissive):**
  - `gregoreesmaa/sheet-music-converter-omr` → `datasets/iter10_5_proxy_2k.tar` (`2,000` train + `64` val full pages, with `2×` dense multi-staff & HW sampling weight)
  - `gregoreesmaa/sheet-music-converter-omr` → `datasets/suite31_hw_eval.tar` (45-piece `suite31_hw` OOD benchmark, 90 pages)

## Summary & Hypothesis
`small_45m` (`~28–31 MB` Hybrid INT4) leaves `~24–27 MB` of our hard `< 55 MB` WebGPU browser budget unused. On dense polyphonic scores and calligraphic historical manuscripts, decoder width (`d_model=384`) and depth (`8` layers) may be capacity-bound when tracking 4+ simultaneous voices, nested beams, and irregular handwritten glyph shapes.

This controlled 2-arm proxy tests whether scaling to `UniOMRv2Config.edge85m()` (`d_model=512`, `n_dec_layers=10`, `n_heads=8`, `n_kv_heads=2`, `max_mem_tokens=3072`) with active `QATInt4ControllerV2` fake-quantization in the final 30% of training unlocks higher dense-page & HW Note F1 **while provably staying `< 55.0 MB` serialized on disk** (real 4-bit packed nibbles + FP16 group-32 scales + FP16 non-Linear weights). Evaluates both RAW and EMA (`decay=0.99` every step) checkpoints plus post-packing dequantized INT4 Note F1.

## Quick Links
- Execution Plan & Decision Tree: [PLAN.md](./PLAN.md)
- Standalone Modal/Colab L4 Runner: [run_colab.py](./run_colab.py)
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-26T22-18-i14-05-edge85m-int4-capacity-proxy.json`
- HF Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-26T22-18-i14-05-edge85m-int4-capacity-proxy.json`
