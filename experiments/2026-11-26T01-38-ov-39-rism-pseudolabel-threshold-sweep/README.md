# `2026-11-26T01-38-ov-39-rism-pseudolabel-threshold-sweep` — Confidence- & Beat-Gated RISM Pseudo-Labeling Threshold Sweep

- **Status:** `COMPLETED — PROMOTE` (Overnight Batch 09, Wave 7 — executed on Kaggle `2× T4`, `99.75 min`, FP32)
- **Baseline:** Commit `164852f` | Checkpoint: HF `checkpoints/iter10_soup_baseline/uni_omr_soup.pt` (`hw_clean 0.2550`, `hw_photo 0.1516`, `printed_clean 0.8895`, `printed_photo 0.8738`, `all45_clean 0.6921`)
- **Single Variable Tested:** FSA-constrained pseudo-label generation on **300 unlabeled archival RISM pages** (`datasets/rism_hw_5707.tar`) followed by a `3 × 3` filtering-threshold sweep (`mean_logprob` >= `{-0.35, -0.20, -0.10}` × `beat_valid_frac` >= `{0.70, 0.85, 0.95}`), measuring pseudo-label yield/quality across all 9 cells AND downstream supervised fine-tuning lift on `suite31_hw` HW (`14` HW + `12` stratified Printed retention check).

## Summary & Hypothesis
Self-training / noisy-student pseudo-labeling on our 5,707 unlabeled archival RISM manuscripts can inject massive real handwritten diversity into `UniOMRNet`, but unfiltered pseudo-labels propagate hallucinated rhythms and pitch drift. Combining **FSA grammar decoding** with **dual gating** — token confidence (`mean_logprob`) AND musical bar-duration conservation (`beat_valid_frac` = `r_rhythm`) — should isolate high-precision archival pages.

**Hypothesis:** High-precision gating (`beat_valid_frac >= 0.85` and `mean_logprob >= -0.20`) will retain a clean, non-trivial subset (`10%–35%` yield) whose downstream fine-tuning produces a strictly larger `suite31_hw` HW Note F1 lift (`>= +0.015`) with less printed drift than looser (`0.70` / `-0.35`) or overly sparse (`0.95` / `-0.10`) thresholds.

## Quick Links
- Execution Plan, Hyperparameters & Decision Tree: [PLAN.md](./PLAN.md)
- Standalone Colab L4 Runner: [run_colab.py](./run_colab.py)
- Outcome: [OUTCOME.md](./OUTCOME.md) (**`PROMOTE`**: `lp-0.35_beat0.70` `26.0%` yield, `hw_avg_delta +0.0236`, `hw_photo +0.0381`, `printed_clean -0.0006`, `printed_photo +0.0033`)
- Hugging Face Eval Target: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-26T01-38-ov-39-rism-pseudolabel-threshold-sweep.json`
- Hugging Face Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-26T01-38-ov-39-rism-pseudolabel-threshold-sweep.json`
