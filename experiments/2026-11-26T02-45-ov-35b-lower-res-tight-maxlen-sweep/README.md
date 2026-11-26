# `2026-11-26T02-45-ov-35b-lower-res-tight-maxlen-sweep` — Lower-Resolution & Tight-`max_len` Boundary Sweep on `suite31_hw`

- **Status:** `QUEUED` (Wave 4 follow-up — successor to [`2026-11-26T01-38-ov-35-maxlen-resolution-inference-sweep`](../2026-11-26T01-38-ov-35-maxlen-resolution-inference-sweep/OUTCOME.md), Google Colab `NVIDIA L4`, `BF16`)
- **Depends On:** `2026-11-26T01-38-ov-35-maxlen-resolution-inference-sweep` (`COMPLETED`)
- **Baseline:** Checkpoint HF `checkpoints/iter10_soup_baseline/uni_omr_soup.pt` — `ov-35` winner `(1280px, 3840)`: `printed_clean 0.8997`, `printed_photo 0.8818`, `hw_clean 0.2799`, `hw_photo 0.1867`, `all45_clean 0.7069`, `all45_photo 0.6655`.
- **Single Variable Tested:** Lower/intermediate inference resolutions `--max-dim in {1024, 1152, 1280, 1408}` crossed with tighter token horizons `--max-len in {2560, 3072, 3584, 3840}` across all **45 `suite31_hw` pieces** (`clean` + `photo` = 90 pages).

## Summary & Hypothesis
`ov-35` proved that `max_len > 3840` strictly hurts `suite31_hw` (0/62 printed pages exceed `2400` tokens; all >3840 pages are HW repetition loops) and that `1280px` beats `1536px` by `+0.0126` on `all45_clean` (`0.7069` vs `0.6942`) and `+0.0181` on `hw_clean` (`0.2799` vs `0.2618`), while `1792px` collapses OOD (`-0.0348`). Because both `1280px` and `3840` sat at the **lower boundary** of `ov-35`, we have not yet bracketed the true peak on the low side.

**Hypothesis:** Sweeping `max_dim in {1024, 1152, 1280, 1408}` × `max_len in {2560, 3072, 3584, 3840}` (16 cells from 4 fast GPU passes at horizon `3840`) will (a) confirm whether `1280px` is the true peak or whether `1152px`/`1408px` gains another `>= +0.003` (especially on `hw_photo` where `1536px` edged out `1280px`), and (b) determine if capping `max_len` at `3072` or `3584` further curbs HW runaway loops without truncating any valid printed or HW notation.

## Quick Links
- Execution Plan & Copy-Paste Colab Commands: [PLAN.md](./PLAN.md)
- Standalone Colab L4 Runner: [run_colab.py](./run_colab.py)
- Predecessor Outcome: [ov-35 OUTCOME.md](../2026-11-26T01-38-ov-35-maxlen-resolution-inference-sweep/OUTCOME.md)
- Outcome: `OUTCOME.md` *(to be written immediately once Colab run completes)*
- Hugging Face Eval Target: `gregoreesmaa/sheet-music-converter-omr` -> `evals/2026-11-26T02-45-ov-35b-lower-res-tight-maxlen-sweep.json`
- Hugging Face Run Ledger: `gregoreesmaa/sheet-music-converter-omr` -> `runs/2026-11-26T02-45-ov-35b-lower-res-tight-maxlen-sweep.json`
