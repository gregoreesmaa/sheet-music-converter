# `2026-11-26T01-38-ov-37-rlvr-kl-group-sweep` — Beat-Conservation + `<sys>`-Loop RLVR `kl_beta` × Group Size (`G`) Sweep

- **Status:** `QUEUED` (Overnight Batch 09, Wave 7 — Google Colab `NVIDIA L4`, `cc=8.9`, BF16)
- **Baseline:** Commit `164852f` | Checkpoint: HF `checkpoints/iter10_soup_baseline/uni_omr_soup.pt` (`hw_clean 0.2550`, `hw_photo 0.1516`, `printed_clean 0.8895`, `printed_photo 0.8738`, `all45_clean 0.6921`)
- **Single Variable Tested:** 2D sweep of label-free GRPO RLVR (`tmp-train/mvd/rlvr_grpo.py`) regularization strength `kl_beta` ∈ `{0.02, 0.06, 0.15}` × rollout group size `G` ∈ `{4, 8}` (6 arms) on unlabeled real historical RISM manuscripts (`datasets/rism_hw_5707.tar`), measuring `suite31_hw` HW Note F1 / beat-conservation gains vs printed retention.

## Summary & Hypothesis
Label-free structural-validity RLVR (`compute_unlabeled_structural_reward`: FSA parse legality + per-measure beat-duration conservation + `<sys>`/EOS completeness + anti-loop penalty) lets `UniOMRNet` learn from 5,707 unlabeled archival RISM handwritten manuscripts without manual transcription. However, unsupervised RLVR carries a classic exploration-vs-drift tradeoff: too-small `kl_beta` (`0.02`) or noisy group advantages (`G=4`) can drift the policy and degrade printed accuracy, whereas too-large `kl_beta` (`0.15`) may freeze HW adaptation.

**Hypothesis:** Increasing group size from `G=4` to `G=8` reduces GRPO advantage variance on hard archival RISM pages, and a moderate KL anchor (`kl_beta=0.06`) achieves the Pareto frontier — yielding `>= +0.015` Note F1 (and `>= +0.05` beat-conservation `r_rhythm`) on `suite31_hw` HW while keeping `printed_clean` / `printed_photo` retention within `<= 0.010` of `uni_omr_soup.pt`.

## Quick Links
- Execution Plan, Hyperparameters & Decision Tree: [PLAN.md](./PLAN.md)
- Standalone Colab L4 Runner: [run_colab.py](./run_colab.py)
- Outcome: `OUTCOME.md` *(to be written immediately once Colab run completes)*
- Hugging Face Eval Target: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-26T01-38-ov-37-rlvr-kl-group-sweep.json`
- Hugging Face Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-26T01-38-ov-37-rlvr-kl-group-sweep.json`
