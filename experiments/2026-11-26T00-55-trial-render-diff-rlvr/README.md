# `2026-11-26T00-55-trial-render-diff-rlvr` — Render-and-Diff Visual Self-Feedback RLVR (60 GRPO Updates)

- **Status:** `RUNNING` (launched `2026-11-26 00:57`, Modal `L4`, BF16)
- **Baseline Commit SHA:** `9d73170`
- **Warm-Start Baseline Checkpoint:** `gregoreesmaa/sheet-music-converter-omr` → `checkpoints/iter10_soup_baseline/uni_omr_soup.pt` (`printed_clean 0.8895`, `printed_photo 0.8738`, `hw_clean 0.2550`, `hw_photo 0.1516`, `all45_clean 0.6921`)

## Summary & Hypothesis
Test **100% label-free Closed-Loop Render-and-Diff Visual Self-Feedback RLVR** (`tmp-train/mvd/trial_2027_render_diff_rlvr.py`) on `UniOMRNet` (`v1`) for `60` GRPO updates (`G=4` rollouts/page under `PrefixState`) using unlabeled sheet-music pages from `datasets/iter10_5_proxy_2k.tar`. Each rollout is converted `bekern -> MusicXML (assembly-serialize) -> SVG (verovio) -> grayscale ink (resvg_py)` and scored against the input page via font/margin-invariant horizontal/vertical ink profiles + multi-scale pooled ink cosine (`0.50`) combined with parse (`0.20`), meter conservation (`0.15`), and anti-loop (`0.15`) rewards.

**Hypothesis:** Even with zero ground-truth labels, visual render-and-diff self-verification penalizes skipped/extra systems, bar-duration drift, and missing note density while `kl_beta=0.04` prevents policy collapse — improving or preserving `suite31_hw` Note F1 and DecodeOK rate over the `uni_omr_soup.pt` baseline and validating the pipeline for future unlabeled RISM / IMSLP post-training.

## Quick Links
- **Plan, Query IDs & Handoff:** [PLAN.md](./PLAN.md)
- **Modal Launcher:** [run_modal.py](./run_modal.py)
- **Outcome:** `OUTCOME.md` *(to be written immediately after Modal run + `suite31_hw` before/after eval completes)*
- **Hugging Face Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Checkpoint: `checkpoints/2026-11-26T00-55-trial-render-diff-rlvr/uni_omr_final.pt`
  - Eval JSON: `evals/2026-11-26T00-55-trial-render-diff-rlvr.json`
  - Run Ledger: `runs/2026-11-26T00-55-trial-render-diff-rlvr.json`
