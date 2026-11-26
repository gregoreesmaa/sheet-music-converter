# `2026-11-26T14-40-eval-northstar-iter12-v2` — UniOMR Iter 12 Stage 2 on OMR-NorthStar

- **Status:** `PLANNED` (ready to launch once Iter 12 Stage 2 finishes step 20,000)
- **Baseline Commit SHA:** `36484e2`
- **Target System:** UniOMR v2 Iter 12 Stage 2 (`step 20,000` EMA: `D:\tmp\mvd-runs\uni-omr-iter12-v2-stage2-20k\uni_omr_v2_ema_best.pt`)
- **Benchmark:** OMR-NorthStar local real-image eval pack (`D:\tmp\northstar-eval-pack\manifest.json`, 125 pages across `CDR`, `ALS`, `RWP`, `HWM` + 4 Pillar 3B Negative Controls)

## Summary & Hypothesis

Establish the first full 4-layer **OMR-NorthStar (`NS`)** headline benchmark score for **UniOMR Iter 12 Stage 2 (`v2` 20k EMA)** and compare it head-to-head against external OMR baselines (`Audiveris`, `oemer`, `homr`) on the exact same 125-page manifest. Because UniOMR Iter 12 is trained on full-page multi-domain data (clean renders, phone-photo degradations, and real historical handwritten manuscripts) with FSA grammar decoding, we expect it to achieve strong `CDR`/`ALS`/`RWP` parity (small `|Δ_photo|`), non-trivial `HWM` transcription where classical/printed-only OMR systems collapse, and low `NMHR` on negative controls.

## Quick Links

- Plan & Commands: [PLAN.md](./PLAN.md)
- Eval Script: [run_eval.py](./run_eval.py)
- Outcome: `OUTCOME.md` *(written immediately upon completion)*
- Shared Scorer: [eval_northstar_runner.py](../../tmp-train/mvd/eval_northstar_runner.py)
- Hugging Face Output Path: `gregoreesmaa/sheet-music-converter-omr` -> `evals/2026-11-26T14-40-eval-northstar-iter12-v2_northstar.json`
