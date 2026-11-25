# `2026-11-26T01-38-ov-38-scribe-ttt-lr-steps-sweep` — SCRIBE/DocTTT Per-Manuscript TTA `steps` × `lr` × `scope` Sweep

- **Status:** `QUEUED` (Overnight Batch 09, Wave 7 — Google Colab `NVIDIA L4`, `cc=8.9`, BF16)
- **Baseline:** Commit `164852f` | Checkpoint: HF `checkpoints/iter10_soup_baseline/uni_omr_soup.pt` (`hw_clean 0.2550`, `hw_photo 0.1516`, `printed_clean 0.8895`, `printed_photo 0.8738`, `all45_clean 0.6921`)
- **Single Variable Tested:** 3D sweep of per-manuscript `SCRIBE/DocTTT` test-time adaptation (`tmp-train/mvd/trial_2028_ttt_adapt.py`) across `steps` ∈ `{2, 4, 8}` × `lr` ∈ `{1e-5, 3e-5, 1e-4}` × `adapt_scope` ∈ `{"encoder_norm_only", "full_encoder"}` (`18` grid cells) on all **14 real historical handwritten manuscripts** in `suite31_hw` (`clean` + `photo` = `28` pages), with bit-exact (`atol=0`) weight reset between pages and trajectories.

## Summary & Hypothesis
Historical handwritten scores vary dramatically in nib width, paper foxing, slant, and staff spacing. While `encoder_norm_only` (`RMSNorm` scales, `< 0.1%` of weights) is ultra-safe against overfitting, it may lack capacity to adapt conv/attention filters to extreme archival hands; conversely, `full_encoder` adaptation may overfit or collapse at `lr=1e-4` / `steps=8`.

**Hypothesis:** A full `3 × 3 × 2` sweep with cumulative stepping will identify the exact `(scope, lr, steps)` sweet spot — specifically testing whether `full_encoder` at conservative `lr` (`1e-5`–`3e-5`) and `steps=4` unlocks `>= +0.025` Note F1 on `suite31_hw` HW over `uni_omr_soup.pt` without collapsing on low-confidence pages.

## Quick Links
- Execution Plan, Hyperparameters & Decision Tree: [PLAN.md](./PLAN.md)
- Standalone Colab L4 Runner: [run_colab.py](./run_colab.py)
- Outcome: `OUTCOME.md` *(to be written immediately once Colab run completes)*
- Hugging Face Eval Target: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-26T01-38-ov-38-scribe-ttt-lr-steps-sweep.json`
- Hugging Face Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-26T01-38-ov-38-scribe-ttt-lr-steps-sweep.json`
