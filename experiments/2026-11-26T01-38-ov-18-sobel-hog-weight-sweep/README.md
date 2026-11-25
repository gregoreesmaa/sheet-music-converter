# `2026-11-26T01-38-ov-18-sobel-hog-weight-sweep`

**Title:** Overnight Exp 18 — MonkeyOCRv2 Sobel-HOG Aux Loss Weight Sweep (`0.0`, `0.05`, `0.15`, `0.30`) for Faint HW Stroke Invariance on `UniOMR v2 small_45m`
**Status:** `QUEUED` (Overnight Batch 02, Wave 5)
**Baseline Commit SHA:** `164852f`
**Target Platform:** Google Colab `L4` (24 GB VRAM, BF16 native, ~35–40 min total, ~3.2 CU)

---

## Summary & Hypothesis

`UniOMR v2` includes a training-only, zero-inference-cost `StrokeHogHead` (`--stroke-hog-weight`, default `0.15`) inspired by MonkeyOCRv2 that predicts 8-bin Sobel orientation histograms from high-resolution `P2` backbone features. Because Sobel orientation histograms are invariant to global ink-darkness scaling, supervising `P2` with Sobel-HOG should prevent faint handwritten pencil/iron-gall strokes from vanishing in early conv layers.

However, the loss weight `--stroke-hog-weight 0.15` has never been swept empirically. Too small (`0.0`) loses stroke-orientation regularization; too large (`0.30`) may over-constrain `P2` features and hurt semantic pitch/rhythm CE convergence.

We run a clean 4-arm sweep on `UniOMR v2 small_45m`:
- **Arm 0 (`hog_000`):** `--stroke-hog-weight 0.0` (disables `StrokeHogHead` cleanly)
- **Arm 1 (`hog_005`):** `--stroke-hog-weight 0.05`
- **Arm 2 (`hog_015`):** `--stroke-hog-weight 0.15` (current default)
- **Arm 3 (`hog_030`):** `--stroke-hog-weight 0.30`

Each arm is evaluated on BOTH standard `proxy_val.jsonl` (64 pages) AND a **Faint-Ink Perturbation Probe** (washed-out contrast / gamma-lifted copies of `proxy_val.jsonl` simulating faint historical iron-gall/pencil manuscripts) to directly measure faint-stroke robustness.

**Hypothesis:** `--stroke-hog-weight 0.05` or `0.15` will achieve the best Pareto point, improving Faint-Ink Probe Token F1 by `>= +1.0 pp` over `0.0` with zero regression on standard `proxy_val`, whereas `0.30` will slightly slow CE convergence.

---

## Quick Links

- **Plan & Runbook:** [PLAN.md](./PLAN.md)
- **Standalone Colab Runner:** [run_colab.py](./run_colab.py)
- **Outcome:** `OUTCOME.md` *(written after Colab run completes)*
- **HF Private Hub Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-26T01-38-ov-18-sobel-hog-weight-sweep.json`
  - Run Ledger: `runs/2026-11-26T01-38-ov-18-sobel-hog-weight-sweep.json`
