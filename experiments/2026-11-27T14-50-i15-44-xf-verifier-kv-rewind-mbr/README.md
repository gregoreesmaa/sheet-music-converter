# `2026-11-27T14-50-i15-44-xf-verifier-kv-rewind-mbr`

- **Status:** `PLANNED` (QUEUED in `experiments/overnight_queue/queue.json`, Wave 0 / P0)
- **Short ID:** `i15-44`
- **Baseline Commit SHA:** `6055c49`
- **Cross-Field Origin:** Compilers (`VerMCTS`/FIM), Drift/RL (`Verifier-Triggered KV-Cache Rewind`), Connectomics (`NEURD`), HITL (`Structure-Conditional MBR`).

## Summary & Hypothesis
Zero-training inference proxy evaluating **Verifier-Triggered KV-Cache Rewind + Structure-Conditional MBR (`kv_rewind_mbr`)** against standard FSA-constrained greedy decoding (`greedy_fsa`) on a challenging 12-page subset of `suite31_hw` (6 printed + 6 real handwritten photo pages) using our best existing Iter 14 checkpoint.

During KV-cached autoregressive decoding, we snapshot the decoder's self-attention KV cache and `grammar.PrefixState` at every bar boundary (`<bos>`, `<sys>`, `bar*`). Whenever a bar closes with a deterministic bar-duration / voice-sync violation (`meter_err > 1e-3` or `voice_err > 1e-3`) or high peak intra-bar entropy (`> 1.6` nats), we **rewind the KV cache in-place to the bar boundary** (reusing precomputed encoder cross-KV for free), generate `K=4` candidate bar continuations (1 peak-entropy top-1 masked greedy + 3 temperature-sampled FSA candidates, max 2 retries/bar), and select the winning bar via **Structure-Conditional MBR** (conservation bonus + log-prob minus asymmetric edit cost weighting structural/header/duration tokens `2×`). We hypothesize this cuts Bar-Sum Violation Rate (`BSVR`) and weighted Fixes-Per-Page (`wFPP`) while improving `Note F1` at modest latency overhead.

## Quick Links
- [PLAN.md](PLAN.md)
- [run_colab.py](run_colab.py)
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-27T14-50-i15-44-xf-verifier-kv-rewind-mbr.json`
- HF Partial Sync: `checkpoints/2026-11-27T14-50-i15-44-xf-verifier-kv-rewind-mbr/`
