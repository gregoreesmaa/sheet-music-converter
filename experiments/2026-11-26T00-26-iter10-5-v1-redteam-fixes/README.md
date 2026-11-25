# `2026-11-26T00-26-iter10-5-v1-redteam-fixes` — Iter 10.5 (`v1` Architecture With All Red-Team & Data Fixes)

- **Status:** `RUNNING` (started `2026-11-26 00:26`, Local RTX 4080 SUPER)
- **Baseline:** Commit `052e210` | Warm-start checkpoint: `D:\tmp\mvd-runs\uni-omr-iter10-v1\uni_omr_soup.pt` (`printed_clean 0.8895`, `printed_photo 0.8738`, `hw_clean 0.2550`, `hw_photo 0.1516`, `all45_clean 0.6921`)
- **Reference Best (Iter 10 `step 12k`):** `printed_clean 0.8993`, `printed_photo 0.8942`, `hw_clean 0.1953`, `hw_photo 0.1376`, `all45_clean 0.6802`

## Summary & Hypothesis
Isolate and validate all 15 silent training, router, augmentation, grammar, and dataset-poison fixes discovered by our Iter-10 Red-Team & Overfit/Underfit audits **on the proven `v1` architecture** before jumping to `UniOMR v2` in Iter 11.

**Hypothesis:** Fixing `tok_dropout + CASS` 32% double-corruption, removing `--struct-tok-weight 2.0` logit distortion, replacing the destructive `5×5` `paper_gate` with edge-connected flood-fill (`99.98%` interior ink preserved), switching to ink-density-adaptive routing (`max_kept_tokens=10,240`), removing 5 GT-destructive augmentations, deduplicating `8,761` manifest rows + capping `real_hw` strip reuse (`220× → 4×`) at `0.06` weight, and adding `0.05` label smoothing will push **BOTH `printed_clean` and `printed_photo` cleanly above `0.91+`** while keeping/lifting **`hw_clean` above `0.26–0.30+`** in `8,000` steps (~90 min).

## Links
- Plan & Query IDs: [PLAN.md](./PLAN.md)
- Outcome: `OUTCOME.md` *(to be written immediately after step-8,000 `suite31_hw` OOD eval completes)*
