# Experiment 5 — `2026-11-26T00-54-ablation-unlabeled-rism-rlvr`

- **Experiment ID:** `2026-11-26T00-54-ablation-unlabeled-rism-rlvr`
- **Status:** `RUNNING` *(launched detached on Modal `L4`; see `PLAN.md` for live `app_id` & URL)*
- **Baseline Commit SHA:** `9d73170`
- **Baseline Checkpoint:** `gregoreesmaa/sheet-music-converter-omr` → `checkpoints/iter10_soup_baseline/uni_omr_soup.pt` (`hw_clean 0.2550`, `printed_clean 0.8895` Strict Note F1 on `suite31_hw`)

## One-Paragraph Summary & Hypothesis

This 1-variable ablation tests **Label-Free Unlabeled-RISM Structural-Validity RLVR (GRPO)** (`tmp-train/mvd/rlvr_grpo.py`) on top of our Iter-10 `uni_omr_soup.pt` checkpoint using our **5,707 unlabeled real historical RISM handwritten manuscript pages** (`datasets/rism_hw_5707.tar`, `unlabeled_frac=1.0`). Rather than requiring human-transcribed ground truth, each rollout group (`G=4`, grammar-constrained) is scored by the deterministic verifier `compute_unlabeled_structural_reward` combining FSA grammar legality + balanced ties/tuplets (`r_parse`, `0.30`), per-measure bar-duration conservation (`r_rhythm`, `0.45`), and completeness / anti-loop / `<eos>` termination (`r_completeness`, `0.25`), regularized toward the frozen Iter-10 reference policy via unbiased `k3` KL (`kl_beta=0.04`). We hypothesize that `120` GRPO updates on real unlabeled RISM pages will improve structural validity, measure rhythm conservation, and Note F1 on the 14 real handwritten `suite31_hw` pieces without regressing printed accuracy.

## Quick Links

- **Execution Plan & Live Query IDs:** [PLAN.md](./PLAN.md)
- **Self-Contained Modal Launcher:** [run_modal.py](./run_modal.py)
- **Outcome & Verdict:** `OUTCOME.md` *(to be written immediately upon completion)*
- **Hugging Face Repo:** `gregoreesmaa/sheet-music-converter-omr`
  - Run ledger: `runs/2026-11-26T00-54-ablation-unlabeled-rism-rlvr.json`
  - Eval report (before vs after 45-piece `suite31_hw`): `evals/2026-11-26T00-54-ablation-unlabeled-rism-rlvr.json`
  - Final checkpoint: `checkpoints/2026-11-26T00-54-ablation-unlabeled-rism-rlvr/uni_omr_rlvr_final.pt`
