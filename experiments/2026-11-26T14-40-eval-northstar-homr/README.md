# `2026-11-26T14-40-eval-northstar-homr` — `homr` (Open-Source Neural OMR) on OMR-NorthStar

- **Status:** `PLANNED`
- **Baseline Commit SHA:** `36484e2`
- **Target System:** `homr` (MIT license — open-source end-to-end neural OMR, installed in an isolated venv `D:\tmp\venvs\homr` or Docker container; extensible to SMT++ via `--backend`)
- **Benchmark:** OMR-NorthStar local real-image eval pack (`D:\tmp\northstar-eval-pack\manifest.json`, 125 pages across `CDR`, `ALS`, `RWP`, `HWM` + 4 Pillar 3B Negative Controls)

## Summary & Hypothesis

Evaluate **`homr`** (a modern permissive MIT-licensed open-source neural OMR system that outputs MusicXML directly from full score images) head-to-head against UniOMR Iter 12, Audiveris, and `oemer` on all 125 pages of our OMR-NorthStar local eval pack. Unlike classical staff-segmentation pipelines (`Audiveris`) or hybrid UNet-assembly (`oemer`), `homr` uses a neural sequence recognizer over detected staves; comparing its `CDR`/`ALS`/`RWP`/`HWM` strata, `Δ_photo`, `wFPP`, and `NMHR` against UniOMR Iter 12 isolates the value of our full-page end-to-end architecture + FSA grammar + real-HW training mix.

## Quick Links

- Plan & Commands: [PLAN.md](./PLAN.md)
- Eval Script: [run_eval.py](./run_eval.py)
- Outcome: `OUTCOME.md` *(written immediately upon completion)*
- Shared Scorer: [eval_northstar_runner.py](../../tmp-train/mvd/eval_northstar_runner.py)
