# `2026-11-26T07-40-i13-30-tok-dropout-exposure-bias-sweep` — Decoder Token Dropout (`--tok-dropout`) Zero-Cost Exposure-Bias Sweep

- **Status:** `QUEUED` (Colab `L4`, Iter 13 Batch 10)
- **Baseline Commit SHA:** `9303e51`
- **Model:** `UniOMR v2 small_45m` (`~41.3M` params)
- **Dataset:** `gregoreesmaa/sheet-music-converter-omr` -> `datasets/iter10_5_proxy_2k.tar` (`2,000` train / `64` val full pages, 100% permissive)

## Audit Finding & Hypothesis
In `tmp-train/mvd/train_uni_v2.py` (line 2250), `--tok-dropout` defaults to `0.05`, which randomly replaces 5% of teacher-forced decoder input tokens with uniform random vocabulary IDs inside `UniOMRv2Net.decode_tokens` (`uni_omr_v2.py:1925-1932`). Unlike CASS (`ov-25`, which explicitly disabled `tok_dropout=0.0` and pays a ~2x decoder forward-pass tax), `--tok-dropout` adds **0% training-time FLOP overhead**, yet it has **never been ablated**!

**Hypothesis:** On long full-page OMR sequences (512-2048 tokens), tuning `--tok-dropout` across `{0.00, 0.05, 0.15, 0.25}` will identify the optimal zero-cost exposure-bias regularizer: either proving `0.15`-`0.25` substantially reduces long-page drift and early EOS (boosting `long > 512` Token F1), or proving that random uniform token corruption hurts clean teacher-forcing convergence and should be lowered to `0.00`.

## Quick Links
- Launcher script: [run_colab.py](./run_colab.py)
- Plan & Decision Tree: [PLAN.md](./PLAN.md)
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` -> `evals/2026-11-26T07-40-i13-30-tok-dropout-exposure-bias-sweep.json`
- HF Run Ledger: `gregoreesmaa/sheet-music-converter-omr` -> `runs/2026-11-26T07-40-i13-30-tok-dropout-exposure-bias-sweep.json`
