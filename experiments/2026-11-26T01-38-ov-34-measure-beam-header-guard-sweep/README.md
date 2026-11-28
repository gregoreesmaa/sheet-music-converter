# `2026-11-26T01-38-ov-34-measure-beam-header-guard-sweep` — Header-Guarded Measure-Beam Search Sweep on `suite31_hw`

- **Status:** `COMPLETED (ITERATE)` (Wave 4 — Overnight Queue Batch 08, Google Colab `NVIDIA L4`, `BF16`, `144.0 min`) — `mbeam2_lim3` hits **`all45_clean=0.7017` (`+0.0077`)**, `printed_clean=0.8966` (`+0.0071`), `hw_clean=0.2700` (`+0.0089`), but `hw_photo` regresses `-0.0177` and overhead is `+81.8%` due to unbounded HW triggers; successor [`2026-11-26T05-35-ov-34b-budgeted-logprob-measure-beam`](../2026-11-26T05-35-ov-34b-budgeted-logprob-measure-beam/) queued.
- **Baseline:** Commit `164852f` | Checkpoint: HF `checkpoints/iter10_soup_baseline/uni_omr_soup.pt` (`printed_clean 0.8895`, `printed_photo 0.8738`, `hw_clean 0.2550`, `hw_photo 0.1516`, `all45_clean 0.6921`)
- **Single Variable Tested:** Violation-triggered measure-scoped beam search width `--measure-beam-size` (`1`, `2`, `4`, `8`) crossed with `MeasureBeamHeaderGuard(const_offset_limit)` (`2`, `3`, `5`) during MTP-4 + FSA-grammar decoding across all **45 `suite31_hw` pieces** (`clean` + `photo` = 90 pages).

## Summary & Hypothesis
Violation-triggered measure-scoped beam search (`_beam_repair_measure` in `tmp-train/mvd/eval_uni_suite31.py`) rewinds the KV cache to the start of a measure whenever an interior bar violates meter or cross-voice duration conservation, and searches for a higher-scoring valid bar (`log_prob - 6.0 * meter_err - 4.0 * voice_err`). Previously, an unguarded measure beam could catastrophically degrade pieces with a misread `time-X-Y` header by forcing every measure to match the wrong meter. The newly added `MeasureBeamHeaderGuard` aborts meter-only repairs once `> const_offset_limit` bars exhibit the exact same non-zero signed meter offset while voices agree.

**Hypothesis:** Header-guarded measure-beam search at `measure_beam_size in {4, 8}` with `const_offset_limit in {2, 3}` will yield a net positive Strict Note F1 gain (`>= +0.005` on `all45_clean` / `all45_photo`) by repairing genuine local rhythm/voice-conservation errors while eliminating header-misread regressions, at `< 25%` decode latency overhead.

## Quick Links
- Execution Plan & Copy-Paste Colab Commands: [PLAN.md](./PLAN.md)
- Standalone Colab L4 Runner: [run_colab.py](./run_colab.py)
- Queue Entry: [batch_08.json](../overnight_queue/batch_08.json)
- Outcome: [OUTCOME.md](./OUTCOME.md) (`ITERATE` — successor [`ov-34b`](../2026-11-26T05-35-ov-34b-budgeted-logprob-measure-beam/))
- Hugging Face Eval Target: `gregoreesmaa/sheet-music-converter-omr` -> `evals/2026-11-26T01-38-ov-34-measure-beam-header-guard-sweep.json`
- Hugging Face Run Ledger: `gregoreesmaa/sheet-music-converter-omr` -> `runs/2026-11-26T01-38-ov-34-measure-beam-header-guard-sweep.json`
