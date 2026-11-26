# `2026-11-26T03-08-ov-36b-multiscale-logprob-fsa-tta` — Clean Multi-Scale TTA + Model Log-Prob & FSA Reranking on Historical HW `suite31_hw`

- **Status:** `QUEUED` (Wave 4 follow-up — successor to [`2026-11-26T01-38-ov-36-hw-multiview-tta-voting`](../2026-11-26T01-38-ov-36-hw-multiview-tta-voting/OUTCOME.md), Google Colab `NVIDIA L4`, `BF16`)
- **Depends On:** `2026-11-26T01-38-ov-36-hw-multiview-tta-voting` (`COMPLETED`)
- **Baseline:** Checkpoint HF `checkpoints/iter10_soup_baseline/uni_omr_soup.pt` — `ov-36` controls: `base_1536` (`hw_clean 0.2611`, `hw_photo 0.1916`), `scale_1280` (`hw_clean 0.2794`, `hw_photo 0.1863`), offline 3-scale `best_fsa_beat` (`hw_clean 0.2954`, `hw_photo 0.1969`).
- **Single Variable Tested:** Replacing destructive `clahe`/`deskew`/`1792px` views with **5 clean in-distribution scale views** (`{1024, 1152, 1280, 1408, 1536}` at `max_len=3840`) and adding **exact teacher-forced length-normalized model log-probability (`mean_logprob`) + entropy (`mean_entropy`)** to `fsa_beat_score` and `mbr_vote_score` reranking across all **14 real historical HW pieces** (`clean` + `photo` = 28 pages).

## Summary & Hypothesis
`ov-36` proved that multi-view TTA has `+0.0419` oracle headroom on `hw_clean` and `+0.0327` on `hw_photo`, but 6-view unsupervised reranking failed because `clahe` (paper-grain amplification) and `deskew` (`±4.00°` desk-border clamp failure on 8/14 `hw_photo` pages) injected net-destructive OOD candidates that poisoned `best_fsa_beat` (`-0.0248` on `hw_photo`). Offline simulation of dropping `clahe`/`deskew` immediately jumped `best_fsa_beat` to **`0.2954` on `hw_clean` (`+0.0343` vs `base_1536`, `+0.0160` vs `scale_1280`)** and **`0.1969` on `hw_photo` (`+0.0053` vs `base_1536`, `+0.0106` vs `scale_1280`)**.

**Hypothesis:** Restricting TTA to 5 clean in-distribution scales (`{1024, 1152, 1280, 1408, 1536}`) at `max_len=3840` and fusing `fsa_beat_score` with exact teacher-forced length-normalized `mean_logprob` (and evaluating the fast 2-view `{1280, 1536}` and 3-view `{1152, 1280, 1536}` Pareto slices) will improve **both** `hw_clean` (`>= +0.020` over `scale_1280` / `>= +0.035` over `base_1536`) and `hw_photo` (`>= +0.015` over `base_1536` with `wins > losses`), clearing the promotion bar for `--tta-scales` in `eval_uni_suite31.py`.

## Quick Links
- Execution Plan & Copy-Paste Colab Commands: [PLAN.md](./PLAN.md)
- Standalone Colab L4 Runner: [run_colab.py](./run_colab.py)
- Predecessor Outcome: [ov-36 OUTCOME.md](../2026-11-26T01-38-ov-36-hw-multiview-tta-voting/OUTCOME.md)
- Outcome: `OUTCOME.md` *(to be written immediately once Colab run completes)*
- Hugging Face Eval Target: `gregoreesmaa/sheet-music-converter-omr` -> `evals/2026-11-26T03-08-ov-36b-multiscale-logprob-fsa-tta.json`
- Hugging Face Run Ledger: `gregoreesmaa/sheet-music-converter-omr` -> `runs/2026-11-26T03-08-ov-36b-multiscale-logprob-fsa-tta.json`
