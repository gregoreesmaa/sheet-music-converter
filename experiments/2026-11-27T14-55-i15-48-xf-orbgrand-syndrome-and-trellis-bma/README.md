# `2026-11-27T14-55-i15-48-xf-orbgrand-syndrome-and-trellis-bma`

- **Status:** `PLANNED` (QUEUED in `experiments/overnight_queue/queue.json`, Wave 0 / P0)
- **Short ID:** `i15-48`
- **Baseline Commit SHA:** `6055c49`
- **Cross-Field Origin:** Information Theory / IDS Coding (`ORBGRAND` / `SOGRAND` IEEE TWC 2024–25 + `Trellis-BMA` multi-view IDS consensus).

## Summary & Hypothesis
Zero-training inference proxy evaluating **ORBGRAND / SOGRAND Syndrome-Driven Ordered-Noise Bar Repair** and **3-View Trellis-BMA Multi-View IDS Consensus** on our best existing checkpoint (`iter14-final` EMA or fallback HF checkpoint) over a fast 12-page hard printed + handwritten subset of `suite31_hw`:
- **Arm 0 (`greedy_fsa`):** Standard FSA-constrained greedy decode recording per-token margin reliabilities (`L_i = log p_1 - log p_2`).
- **Arm 1 (`orbgrand_sogrand`):** Whenever a decoded bar violates musical duration/BSVR conservation (`syndrome != 0`), runs fast CPU **ORBGRAND** ordered-noise search over the lowest-margin token positions (up to 64 logistic-weight noise patterns per bar, flipping low-margin duration/dot/rest tokens to their top-2/top-3 FSA-valid alternatives) until `syndrome == 0`, and computes **SOGRAND**'s analytical "not-found" posterior `P_not_found` as a calibrated bar-level uncertainty score.
- **Arm 2 (`trellis_bma_3view`):** On flagged/high-syndrome pages (`P_not_found > tau` or residual `syndrome != 0`), decodes 3 cheap contrast/scale views (`base_1408`, `clahe_1408`, `scale_1280`) and fuses bar-aligned streams via **Trellis-BMA** (Insertion-Deletion-Substitution bar-aligned majority consensus with conservation tie-breaking).

**Hypothesis:** Unlike blind beam search, ORBGRAND exploits exact algebraic beat-conservation syndromes to repair duration/dot errors in microseconds on CPU, while SOGRAND's analytical `P_not_found` selectively routes only genuinely ambiguous pages to 3-view Trellis-BMA consensus, improving Token/Note F1 and slashing bar-syndrome violations at minimal latency overhead.

## Quick Links
- [PLAN.md](PLAN.md)
- [run_colab.py](run_colab.py)
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-27T14-55-i15-48-xf-orbgrand-syndrome-and-trellis-bma.json`
- HF Partial Sync: `checkpoints/2026-11-27T14-55-i15-48-xf-orbgrand-syndrome-and-trellis-bma/`
