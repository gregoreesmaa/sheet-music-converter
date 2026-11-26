# `2026-11-26T14-40-eval-northstar-audiveris` — Audiveris (Black-Box Docker) on OMR-NorthStar

- **Status:** `PLANNED`
- **Baseline Commit SHA:** `36484e2`
- **Target System:** Audiveris 5.3.x (**AGPL-3.0** — strictly isolated black-box Docker CLI container ONLY; NEVER imported, linked, or used in training/inference paths)
- **Benchmark:** OMR-NorthStar local real-image eval pack (`D:\tmp\northstar-eval-pack\manifest.json`, 125 pages across `CDR`, `ALS`, `RWP`, `HWM` + 4 Pillar 3B Negative Controls)

## Summary & Hypothesis

Measure classical/hybrid symbolic OMR (**Audiveris**) head-to-head against UniOMR Iter 12 on the identical 125-page OMR-NorthStar pack. Audiveris is strong on clean digital engravings (`CDR`) and moderate archival scans (`ALS`), but classical staff-line / glyph-segmentation pipelines degrade sharply under phone-photo warp/lighting (`RWP`, large negative `Δ_photo`) and collapse on real historical handwritten manuscripts (`HWM`), exposing why OMR-NorthStar's geometric-mean + worst-stratum composite `NS` penalizes single-domain specialists.

## License Hygiene (CRITICAL)

- Audiveris is **AGPL-3.0**. Per `AGENTS.md`, it MUST NEVER enter `sheet-music-converter`'s training, data generation, or production inference pipeline.
- `run_eval.py` invokes Audiveris strictly as an ephemeral, network-disabled black-box Docker container (`docker run --rm --network none ...`), reading only the exported `.mxl`/`.musicxml` files for benchmark scoring.

## Quick Links

- Plan & Commands: [PLAN.md](./PLAN.md)
- Eval Script: [run_eval.py](./run_eval.py)
- Outcome: `OUTCOME.md` *(written immediately upon completion)*
- Shared Scorer: [eval_northstar_runner.py](../../tmp-train/mvd/eval_northstar_runner.py)
