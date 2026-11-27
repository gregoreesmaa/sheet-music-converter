# `2026-11-27T14-15-i15-14-3axis-structural-dec-rope`

- **Status:** `PLANNED` (QUEUED in `experiments/overnight_queue/queue.json`, Wave 1)
- **Short ID:** `i15-14`
- **Baseline Commit SHA:** `a32c810`

## Summary & Hypothesis
Tests **3-Axis Structural Decoder RoPE (`StructuralRoPE3D`)** in the causal decoder self-attention of `UniOMRv2Net(edge85m, ax0_gl6, d=512)` against stock 1D sequential RoPE (`ctrl`) under a controlled 2-arm 600-step BF16 Muon+AdamW proxy on `iter10_5_proxy_2k.tar`.

Instead of 1D sequential token positions, `StructuralRoPE3D` splits `head_dim=64` into three rotary subspaces `(16, 16, 32)` for `(system_idx, measure_idx, intra_measure_pos)` derived **100% causally** from the prefix tokens (`<sys>` increments `system_idx` and resets `measure_idx` + `intra_measure_pos`; barline tokens `bar`, `bar-light-heavy`, `bar-repeat-left`, `bar-repeat-right`, `bar-double` increment `measure_idx` and reset `intra_measure_pos`). Gives O(1) structural alignment across staves/measures, prevents cross-system drift/loops, and costs **`0.000 MB` INT4**.

## Quick Links
- [PLAN.md](PLAN.md)
- [run_colab.py](run_colab.py)
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-27T14-15-i15-14-3axis-structural-dec-rope.json`
- HF Partial Sync: `checkpoints/2026-11-27T14-15-i15-14-3axis-structural-dec-rope/`
