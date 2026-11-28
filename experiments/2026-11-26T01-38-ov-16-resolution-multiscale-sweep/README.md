# `2026-11-26T01-38-ov-16-resolution-multiscale-sweep`

**Title:** Overnight Exp 16 — Full-Page Resolution & Multi-Scale Sweep (`1024px` vs `1280px` vs `1536px` Multi-Scale) on `UniOMR v2 small_45m`
**Status:** `COMPLETED` — Verdict: **`DISCARD`** (see [OUTCOME.md](./OUTCOME.md))
**Baseline Commit SHA:** `164852f`
**Target Platform:** Google Colab `L4` (24 GB VRAM, BF16 native, ~35–40 min total, ~3.2 CU)

---

## Summary & Hypothesis

`UniOMR v2` (`AnisotropicMusicBackbone` + `HybridAxialMergerEncoder` + `InkTokenRouterV2` + `AspectNormalizedRoPE2D`) was specifically engineered to scale beyond `1280px` to `1536px` while keeping routed global memory tokens bounded via `InkTokenRouterV2` (`keep_ratio=0.40`, `dynamic_routing=True`). However, we have not yet measured the exact accuracy / token-budget / throughput Pareto frontier across full-page input resolutions on `UniOMR v2 small_45m`:

1. **Arm A (`1024px_fixed`):** `--fullpage-max-dim 1024`, fixed scale (`multi_scale=False`). Fastest baseline; tests whether tiny accidentals, ledger lines, and augmentation dots degrade below `1280px`.
2. **Arm B (`1280px_fixed`):** `--fullpage-max-dim 1280`, fixed scale (`multi_scale=False`). Current default full-page resolution.
3. **Arm C (`1536px_multiscale`):** `--fullpage-max-dim 1536`, `MultiScaleBatchSampler` enabled (`multi_scale=True`, `prefetch_threads=0`, `num_workers=2`). Tests whether higher spatial fidelity + multi-scale invariance improves held-out `proxy_val` Token F1 / SER / `sys_count_exact` without blowing up routed token count or wall-clock speed.

**Hypothesis:** `1536px_multiscale` will deliver `>= +1.5 pp` higher held-out `proxy_val` Token F1 (and lower SER) than `1024px_fixed` and `>= +0.5 pp` over `1280px_fixed` with `< 1.45x` step-time overhead because `InkTokenRouterV2` keeps routed tokens sub-linear in pixel area and `AspectNormalizedRoPE2D` preserves rotary phase invariance across scales.

---

## Quick Links

- **Plan & Runbook:** [PLAN.md](./PLAN.md)
- **Standalone Colab Runner:** [run_colab.py](./run_colab.py)
- **Outcome:** [OUTCOME.md](./OUTCOME.md) (`DISCARD` — `1024px_fixed` `0.2305` / `1280px_fixed` `0.2301` / `1536px_multiscale` `0.2291` EMA F1; saturated at `eval_max_len=512` + `450`-step floor)
- **HF Private Hub Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-26T01-38-ov-16-resolution-multiscale-sweep.json`
  - Run Ledger: `runs/2026-11-26T01-38-ov-16-resolution-multiscale-sweep.json`
