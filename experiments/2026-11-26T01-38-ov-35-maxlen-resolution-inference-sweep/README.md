# `2026-11-26T01-38-ov-35-maxlen-resolution-inference-sweep` — Inference `--max-len` x Input Resolution Sweep on `suite31_hw`

- **Status:** `COMPLETED (ITERATE)` (Wave 4 — Overnight Queue Batch 08, Google Colab `NVIDIA L4`, `BF16`; successor [`2026-11-26T02-45-ov-35b-lower-res-tight-maxlen-sweep`](../2026-11-26T02-45-ov-35b-lower-res-tight-maxlen-sweep/README.md) queued)
- **Baseline:** Commit `164852f` | Checkpoint: HF `checkpoints/iter10_soup_baseline/uni_omr_soup.pt` (`1536px`, `max_len=4096`: `printed_clean 0.8895`, `printed_photo 0.8738`, `hw_clean 0.2550`, `hw_photo 0.1516`, `all45_clean 0.6921`)
- **Single Variable Tested:** Inference-time token horizon `--max-len` (`3840`, `5120`, `6144`) crossed with input resolution `--max-dim` (`1280px`, `1536px`, `1792px`) across all **45 `suite31_hw` pieces** (`clean` + `photo` = 90 pages).

## Summary & Hypothesis
Dense multi-system full pages (orchestral/piano scores and dense historical manuscripts) can either suffer from **token truncation** when `max_len` cuts off the bottom system(s) or from **fine-detail loss** (ledger lines, accidentals, 16th-note beams, tight handwritten ink) when downsampled to `1280px` / `1536px`. Conversely, higher resolution (`1792px`) may push RoPE / router distributions slightly out-of-domain if trained primarily at `1536px`, while longer `max_len` (`5120`/`6144`) could expose repetition-loop pathologies if not properly gated.

**Hypothesis:** Isolating `max_len in {3840, 5120, 6144}` x `max_dim in {1280, 1536, 1792}` will cleanly separate (a) truncation-limited pieces that gain `>= +0.01` Note F1 from `5120`/`6144` tokens vs (b) resolution-limited printed/HW pieces that gain from `1792px` (or improve speed at `1280px` with minimal accuracy drop).

## Quick Links
- Execution Plan & Copy-Paste Colab Commands: [PLAN.md](./PLAN.md)
- Standalone Colab L4 Runner: [run_colab.py](./run_colab.py)
- Queue Entry: [batch_08.json](../overnight_queue/batch_08.json)
- Outcome: [OUTCOME.md](./OUTCOME.md) (`1280px / max_len=3840` wins `all45_clean=0.7069` [`+0.0126` vs `1536px`], `hw_clean=0.2799` [`+0.0181`], `printed_clean=0.8997` [`+0.0102`]; `max_len > 3840` strictly harmful)
- Hugging Face Eval Target: `gregoreesmaa/sheet-music-converter-omr` -> `evals/2026-11-26T01-38-ov-35-maxlen-resolution-inference-sweep.json`
- Hugging Face Run Ledger: `gregoreesmaa/sheet-music-converter-omr` -> `runs/2026-11-26T01-38-ov-35-maxlen-resolution-inference-sweep.json`
