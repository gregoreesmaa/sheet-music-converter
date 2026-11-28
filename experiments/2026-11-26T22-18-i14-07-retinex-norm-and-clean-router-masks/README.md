# 2026-11-26T22-18-i14-07-retinex-norm-and-clean-router-masks

- **Experiment ID:** `2026-11-26T22-18-i14-07-retinex-norm-and-clean-router-masks`
- **Wave / Short ID:** Wave 11 / `i14-07` (Safe — Theorist 4 `ALS` Cure)
- **Status:** `COMPLETED (DISCARD)`
- **Baseline Commit SHA:** `87b1ec8`
- **Baseline Checkpoint:** `checkpoints/iter12_v2_best/uni_omr_v2_ema_best.pt` (fallback `checkpoints/iter11_v2_best/uni_omr_v2_ema_best.pt`)

## Summary & Hypothesis

Forensic analysis of `ALS` (Archival Low-Contrast Sepia scans, currently `51.13` Note F1 vs target `92+`) showed that low-contrast parchment, dark book gutters, and show-through bleed-through fool `InkTokenRouterV2._compute_ink_and_paper_gate` into treating gutters and bleed-through as ink and evicting real notation tokens. Furthermore, `InkTokenRouterV2` currently supervises its saliency head against the degraded input's own heuristic `local_ink_grid`.

**Hypothesis:** Prepending a `< 0.5M`-parameter (`~0.14M`) differentiable **Retinex Background Normalizer** ($I_{\text{norm}} = I / (B(x,y) + \epsilon)$ with a zero-init residual gate for exact step-0 identity) AND supervising `InkTokenRouterV2`'s `saliency_logits` against the **pre-degradation clean-render ink mask** (so gutters/stains/bleed-through get target `0.0` while true notation gets target `1.0`) will dramatically raise router notation recall on sepia/gutter/bleed-through scans and lift `ALS` / `hw_photo` Note F1 without hurting clean printed pages.

## Quick Links

- Plan: [PLAN.md](PLAN.md)
- Runner: [run_colab.py](run_colab.py)
- Outcome: [OUTCOME.md](OUTCOME.md)
- HF Eval JSON: `evals/2026-11-26T22-18-i14-07-retinex-norm-and-clean-router-masks.json`
- HF Run JSON: `runs/2026-11-26T22-18-i14-07-retinex-norm-and-clean-router-masks.json`
