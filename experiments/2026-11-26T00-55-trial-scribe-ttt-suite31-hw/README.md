# `2026-11-26T00-55-trial-scribe-ttt-suite31-hw` — SCRIBE / DocTTT Per-Manuscript Test-Time Adaptation on `suite31_hw`

- **Status:** `RUNNING` (launched `2026-11-26 00:58` on Modal `NVIDIA L4`, `cc=8.9`, BF16 — App ID `ap-x3HATdhn6O1gLztsLTSnVO`)
- **Baseline:** Commit `9d73170` | Checkpoint: HF `checkpoints/iter10_soup_baseline/uni_omr_soup.pt` (`hw_clean 0.2550`, `hw_photo 0.1516`, `printed_clean 0.8895`, `printed_photo 0.8738`, `all45_clean 0.6921`)
- **Single Variable Tested:** Per-manuscript on-device Test-Time Training (`tmp-train/mvd/trial_2028_ttt_adapt.py`, `TTTSession` adapting `< 0.1%` of weights — `RMSNorm` scales in `backbone`/`enc_layers`/`enc_norm`) for `3` and `5` gradient steps per unseen historical handwritten manuscript, with bit-exact (`atol=0`) weight reset between pages.

## Summary & Hypothesis
Unseen historical handwritten manuscripts exhibit severe writer-specific ink, nib, spacing, and staff-curvature domain shift relative to printed and synthetic HW training pages, causing `uni_omr_soup.pt` to score `hw_clean 0.2550` / `hw_photo 0.1516` despite `0.89+` on printed scores.

**Hypothesis:** Running `3` or `5` low-LR (`5e-5`) self-supervised TTT steps per manuscript combining (1) Multi-View Encoder & Router Consistency and (2) Confidence-Gated (`prob >= 0.85`, loop-filtered) Conditional Entropy Minimization anchored via KL + L2 to the pre-TTT snapshot will improve `suite31_hw` Strict Note F1 on the **14 unseen historical handwritten pieces** (`clean` and `photo` = 28 pages) by `>= +0.02` without hurting decode validity or leaking state across documents.

## Quick Links
- Execution Plan, Live Modal Query IDs & Commands: [PLAN.md](./PLAN.md)
- Modal Runner Script: [run_modal.py](./run_modal.py)
- Outcome: `OUTCOME.md` *(to be written immediately once Modal run completes)*
- Hugging Face Eval Target: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-26T00-55-trial-scribe-ttt-suite31-hw.json`
- Hugging Face Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-26T00-55-trial-scribe-ttt-suite31-hw.json`
