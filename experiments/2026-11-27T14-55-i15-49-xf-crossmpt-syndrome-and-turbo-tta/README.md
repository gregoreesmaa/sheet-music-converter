# `2026-11-27T14-55-i15-49-xf-crossmpt-syndrome-and-turbo-tta`

- **Status:** `PLANNED` (QUEUED in `experiments/overnight_queue/queue.json`, Wave 0 / P0)
- **Short ID:** `i15-49`
- **Baseline Commit SHA:** `6055c49`
- **Cross-Field Origin:** Channel Coding (`CrossMPT` ICLR 2025 syndrome-masked cross-attention + Zero-Label Syndrome-Driven Turbo Equalization TTA).

## Summary & Hypothesis
Controlled 2-arm 600-step BF16 Muon+AdamW proxy on `UniOMRv2Net(edge85m, ax0_gl6, d=512)` using `iter10_5_proxy_2k.tar`:
- **Arm 0 (`ctrl`):** Standard `edge85m + ax0_gl6`.
- **Arm 1 (`crossmpt_turbo`):**
  1. Attaches a tiny 2-layer **`CrossMPTRefiner`** (`< 1.5 MB` INT4) performing bar-masked cross-attention between token logit-magnitude/hidden features and per-bar differentiable conservation syndrome embeddings (`expected_bar_duration - bar_capacity`), emitting a gated residual logit correction.
  2. Evaluates **3-step zero-label syndrome-minimizing encoder-norm Turbo TTA** on degraded validation pages (adapting only encoder normalization affine parameters to minimize expected bar-conservation syndrome + entropy, then restoring weights per page).

**Hypothesis:** Standard causal decoders commit local duration/dot errors without global bar-level feedback. Coupling token logit magnitudes to explicit bar-syndrome embeddings via `CrossMPT` cross-attention lets the model self-correct conservation violations in a single forward pass (`< 1.5 MB` INT4, total `< 55.0 MB`), and 3-step zero-label syndrome-driven encoder-norm TTA further adapts to degraded phone-photo pages without ground-truth labels.

## Quick Links
- [PLAN.md](PLAN.md)
- [run_colab.py](run_colab.py)
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-27T14-55-i15-49-xf-crossmpt-syndrome-and-turbo-tta.json`
- HF Partial Sync: `checkpoints/2026-11-27T14-55-i15-49-xf-crossmpt-syndrome-and-turbo-tta/`
