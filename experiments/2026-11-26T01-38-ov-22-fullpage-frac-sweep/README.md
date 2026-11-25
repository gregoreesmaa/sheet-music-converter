# `2026-11-26T01-38-ov-22-fullpage-frac-sweep` — `fullpage_frac` Curriculum Sweep (`0.70` vs `0.85` vs `1.00`) on `UniOMR v2 small_45m`

- **Status:** `PLANNED` (Overnight Queue Batch 04, Wave 5 — `experiments/overnight_queue/batch_04.json`)
- **Baseline Commit SHA:** `164852f`
- **Model:** `UniOMR v2 small_45m` (`~43M` params, Anisotropic FPN + Hybrid Axial/Merger/2D-RoPE Encoder + DeepStack GQA MTP-4 Decoder)
- **Dataset:** `gregoreesmaa/sheet-music-converter-omr` -> `datasets/iter10_5_proxy_2k.tar` (`2,000` train + `64` val full pages, 100% permissive, 0 `suite31_hw` overlap) + deterministically sliced aligned system strips derived from `proxy_train.jsonl`

## Summary & Hypothesis
1-variable controlled curriculum ablation for Iter-11 `UniOMR v2` training. All three arms run `800` steps from identical scratch initialization (`seed=42`) on `datasets/iter10_5_proxy_2k.tar` with identical `B=2 GA=4` (`eff=8`), `strip_batch_size=4`, `fullpage_max_dim=1280`, `max_seq_len=1536`, native `BF16`, `--optimizer muon`, and WSD schedule (`lr=3e-4`, `warmup=80`, `wsd_decay_frac=0.25`, `min_lr_ratio=0.05`). **The ONLY variable changed is `--fullpage-frac` (`0.70` vs `0.85` vs `1.00`):**
- **Arm A (`fp_070`):** `--curriculum mixed --fullpage-frac 0.70` (70% full-page steps, 30% single-system strip steps)
- **Arm B (`fp_085`):** `--curriculum mixed --fullpage-frac 0.85` (85% full-page steps, 15% single-system strip steps)
- **Arm C (`fp_100`):** `--curriculum fullpage --fullpage-frac 1.00` (100% pure full-page steps, 0% strip steps)

**Hypothesis:** Keeping a modest `15%` single-system strip anchor (`--fullpage-frac 0.85`) stabilizes local note/rhythm decoding and improves held-out full-page `proxy_val` Token F1 / SER over `0.70` (too much strip budget, fewer full-page layout updates) while preventing the early alignment drift that can appear under `1.00` pure full-page training — or conversely, if `UniOMR v2`'s Anisotropic FPN + Axial/2D-RoPE encoder already solves full-page layout cleanly, `1.00` will win by devoting 100% of compute to target-distribution full pages.

## Quick Links
- Self-contained Colab L4 runner: [run_colab.py](./run_colab.py)
- Experiment Plan & Decision Tree: [PLAN.md](./PLAN.md)
- Queue Manifest: [batch_04.json](../overnight_queue/batch_04.json)
- Outcome: `OUTCOME.md` *(to be written immediately after HF eval JSON is retrieved)*
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` -> `evals/2026-11-26T01-38-ov-22-fullpage-frac-sweep.json`
- HF Run Ledger: `gregoreesmaa/sheet-music-converter-omr` -> `runs/2026-11-26T01-38-ov-22-fullpage-frac-sweep.json`
