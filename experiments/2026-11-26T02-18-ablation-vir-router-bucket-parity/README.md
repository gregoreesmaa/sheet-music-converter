# `2026-11-26T02-18-ablation-vir-router-bucket-parity` — Un-Confounded `ViRTwoTierRouter` Bucket & Token-Budget Parity Ablation

- **Status:** `QUEUED` (in `experiments/overnight_queue/queue.json`, `depends_on: ["2026-11-26T00-54-ablation-vir-vico-router"]`)
- **Predecessor:** [`2026-11-26T00-54-ablation-vir-vico-router`](../2026-11-26T00-54-ablation-vir-vico-router/OUTCOME.md) (`ITERATE`)
- **Platform:** Google Colab `1× NVIDIA L4` (`24 GB`, native BF16, ~28–34 min, ~2.5 CU)
- **Dataset:** `gregoreesmaa/sheet-music-converter-omr` → `datasets/iter10_5_proxy_2k.tar` (`2,000` train + `64` val full pages, zero `suite31_hw` overlap)

## Summary & Hypothesis
In Exp 3 (`2026-11-26T00-54-ablation-vir-vico-router`), `ViRTwoTierRouter+ViCO` lost to `InkTokenRouterV2` (`-6.04 pp` raw F1, `-8.40 pp` EMA F1, `-6.28 pp` `ink_recall`), so **`InkTokenRouterV2` is kept for Iter 11**. Forensic scrutiny revealed **two concrete confounders** in Exp 3:
1. **`DEFAULT_STATIC_BUCKETS` ceiling at `2048`:** Hard-truncated `k_fine` to `2048` (`n_out <= 2144`) on `1280px` pages (`mean_n_in = 5102.5`), starving `ink_recall` (`0.9030` vs `0.9658`).
2. **Training token-budget mismatch (`0.40` vs `0.75`):** `InkTokenRouterV2` kept `0.75 * N` (up to `3068`) during training while `ViRTwoTierRouter` kept `0.40 * N` (`1536` or `2048`), giving `baseline` ~30% more visual tokens per training step.

This 3-arm follow-up (`600` steps/arm, `seed=42`, `lr=3e-4`, `Muon+AdamW`, `dim=1280`, BF16, `ema_decay=0.99` matched to 600 steps) removes both confounders and separates `ViRTwoTierRouter` from `ViCO`:
1. **`baseline_ink_router`**: `InkTokenRouterV2` (`max_keep_ratio=0.75`, `max_mem_tokens=3072`) — Iter 11 reference.
2. **`vir_parity_no_vico`**: `ViRTwoTierRouter` with `fine_buckets=(256, 512, 768, 1024, 1536, 2048, 2560, 2976)`, `keep_ratio=0.75` during training (matching `InkTokenRouterV2`), dynamic routing at eval (`0.15..0.75`), `k_coarse=96`, **`vico_weight=0.0`**.
3. **`vir_parity_vico005`**: Same bucket-parity `ViRTwoTierRouter` + gentle ViCO (`vico_weight=0.05`, `vico_every=2`, dense teacher `0.85`).

**Hypothesis:** Restoring the `2560 / 2976` static buckets and `0.75` training token-budget parity will recover `ink_recall >= 0.96`, revealing cleanly whether `ViRTwoTierRouter` (with or without ViCO) beats `InkTokenRouterV2` for Iter 12 or should be permanently discarded.

## Quick Links
- Launcher: [run_colab.py](./run_colab.py)
- Plan & Decision Tree: [PLAN.md](./PLAN.md)
- Predecessor Outcome: [2026-11-26T00-54-ablation-vir-vico-router/OUTCOME.md](../2026-11-26T00-54-ablation-vir-vico-router/OUTCOME.md)
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-26T02-18-ablation-vir-router-bucket-parity.json`
