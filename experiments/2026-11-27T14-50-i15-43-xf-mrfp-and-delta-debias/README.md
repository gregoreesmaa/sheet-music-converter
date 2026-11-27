# `2026-11-27T14-50-i15-43-xf-mrfp-and-delta-debias`

- **Status:** `PLANNED` (Wave 0 P0 cross-field proxy)
- **Short ID:** `i15-43`
- **Baseline Commit SHA:** `6055c49`
- **Cross-Field Origin:** Sim2Real (`MRFP`, CVPR 2024 — parameter-free Multi-Resolution Feature Perturbation) + Medical Pathology (`DELTA`, 2026 — counterfactual nuisance-subspace projection).

## Summary & Hypothesis
Tests a zero-inference-cost (`+0.000 MB` INT4, `0%` extra inference FLOPs) two-stage sim-to-real & domain-debiasing pipeline on `UniOMRv2Net(edge85m, ax0_gl6, d=512)` against the stock control (`ctrl`) under a controlled 2-arm 600-step BF16 Muon+AdamW proxy on `iter10_5_proxy_2k.tar`:

1. **Training-time parameter-free `MRFP`:** Perturbs fine-scale channel statistics (`stage1 /2`, `stage2 /4`) and coarse-scale feature style (`stage3 /8`, `stage4 /8`) across multi-scale `AnisotropicMusicBackbone` stages during training only, annealed linearly from `1.0 -> 0.0` during the final 30% of training (steps `420..600`). Forces the encoder to rely on stroke/staff topology rather than synthetic engraving/paper appearance statistics.
2. **Post-training `DELTA` counterfactual nuisance-subspace projection:** After training, extracts paired encoder `mem_final` representations on clean vs. synthetically degraded validation pages, estimates the top-`k` synthetic-appearance nuisance subspace `U_k` via SVD of `H_deg - H_clean`, and folds the orthogonal projector `P_k = I - U_k U_k^T` directly into decoder cross-attention `k_proj` and `v_proj` weights (`W' = W @ P_k`) at `+0.000 MB` and `0%` latency cost. Evaluates both clean and degraded validation CE before and after `DELTA` folding (and after Hybrid INT4 packing).

## Quick Links
- [PLAN.md](PLAN.md)
- [run_colab.py](run_colab.py)
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-27T14-50-i15-43-xf-mrfp-and-delta-debias.json`
- HF Partial Sync: `checkpoints/2026-11-27T14-50-i15-43-xf-mrfp-and-delta-debias/`
