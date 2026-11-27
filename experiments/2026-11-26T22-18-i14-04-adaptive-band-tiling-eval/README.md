# `2026-11-26T22-18-i14-04-adaptive-band-tiling-eval` — Zero-Retrain Adaptive Band-Tiling Probe (Safe)

- **Experiment ID:** `2026-11-26T22-18-i14-04-adaptive-band-tiling-eval`
- **Short ID / Wave:** `i14-04` (Wave 11 — Optical Resolution & Token-Cap Wall)
- **Status:** `COMPLETED (ITERATE → i14-04b)` — see [OUTCOME.md](./OUTCOME.md)
- **Baseline Commit SHA:** `87b1ec8`
- **Platform / GPU:** Modal `1× NVIDIA L4` (`24 GB` VRAM, BF16 native, ~15 min, ~1.2 CU; Colab L4 compatible)
- **Model Footprint:** `0` new parameters — evaluates frozen `UniOMR v2 small_45m` checkpoint (`~42M` infer params, `< 32 MB` Hybrid FP16/INT4)
- **Dataset (100% permissive):**
  - `gregoreesmaa/sheet-music-converter-omr` → `datasets/suite31_hw_eval.tar` (45-piece `suite31_hw` OOD benchmark: 31 printed + 14 real historical HW, `clean` + `photo` = 90 pages)

## Summary & Hypothesis
Shrinking dense 24–32 staff pages to `1408px` leaves only `~4.7 px` per staff-space (`~1.5 px` for augmentation dots and 32nd-note beam gaps — physically aliased), while `max_mem_tokens=3072` drops `20–62%` of ink tokens on dense pages. Unlike `i13-19-coarse2fine-zoom-omr` (which trains a learned 2-pass `SystemBandProposalHead` and decodes single-system strips), this **zero-retrain** probe tests whether pure inference-time **adaptive multi-system band-tiling** unlocks immediate `Strict Note F1` gains on dense printed & handwritten `suite31_hw` pages:

1. Estimate effective `staff_space_px` at `1408px` (via horizontal ink-projection autocorrelation) and system count `n_sys` from the native image.
2. Trigger tiling **only** on dense pages (`staff_space_px < 7.0 px` OR `n_sys >= 5`); sparse pages run standard single-pass `1408px` with zero latency overhead.
3. Split dense pages along the deepest inter-system whitespace valleys into overlapping **2–3 system bands** (`±4%` vertical padding), rendered at native `>= 10 px/staff-space` (up to `1536px` width) so every band stays well under `max_mem_tokens=3072` with `~100%` ink recall.
4. Decode each band with the frozen checkpoint, deduplicate boundary overlap systems, and **stitch with header carry-forward** (propagating clef/key/time-signature tokens from band 0 if omitted in later bands).

Compares 3 inference arms apples-to-apples:
- **Arm A (Control):** Standard single-pass `1408px` (`max_mem_tokens=3072`).
- **Arm B (High-Res Single-Pass):** Single-pass `1792px` (`max_mem_tokens=3072` — isolates whether optical resolution alone helps or is still starved by the `3072` token cap).
- **Arm C (Treatment):** Adaptive 2–3 system band-tiling (`>= 10 px/staff-space`) with overlap deduplication & header carry-forward.

## Quick Links
- Execution Plan & Decision Tree: [PLAN.md](./PLAN.md)
- Standalone Modal/Colab L4 Runner: [run_colab.py](./run_colab.py)
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-26T22-18-i14-04-adaptive-band-tiling-eval.json`
- HF Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-26T22-18-i14-04-adaptive-band-tiling-eval.json`
