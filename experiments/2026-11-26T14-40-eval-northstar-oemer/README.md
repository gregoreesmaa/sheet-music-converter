# `2026-11-26T14-40-eval-northstar-oemer` — `oemer` (Black-Box CLI) on OMR-NorthStar

- **Status:** `DONE` (`NS = 0.79 [0.41, 1.05]`, `Tier-W = 3.09`, `NMHR = 0.0%` — published to `omr-northstar` Public Leaderboard)
- **Baseline Commit SHA:** `36484e2`
- **Target System:** `oemer` (MIT license — open-source UNet + symbolic-assembly OMR, invoked via isolated black-box CLI subprocess)
- **Benchmark:** OMR-NorthStar local real-image eval pack (`D:\tmp\northstar-eval-pack\manifest.json`, 125 pages across `CDR`, `ALS`, `RWP`, `HWM` + 4 Pillar 3B Negative Controls)

## Summary & Hypothesis

Evaluate **`oemer`** head-to-head on all 125 pages of our OMR-NorthStar local eval pack (`CDR`, `ALS`, `RWP`, `HWM` + 4 negative controls). `oemer` combines UNet semantic segmentation with classical music-assembly heuristics; on `suite31` printed clean it previously achieved moderate note recall with fragile measure-level rhythm/barlines, and dropped sharply on phone photos (`RWP`) and handwritten manuscripts (`HWM`). Measuring its full 4-layer `NS` scorecard provides an apples-to-apples permissive open-source neural-hybrid baseline.

## Quick Links

- Plan & Commands: [PLAN.md](./PLAN.md)
- Eval Script: [run_eval.py](./run_eval.py)
- Outcome: `OUTCOME.md` *(written immediately upon completion)*
- Shared Scorer: [eval_northstar_runner.py](../../tmp-train/mvd/eval_northstar_runner.py)
