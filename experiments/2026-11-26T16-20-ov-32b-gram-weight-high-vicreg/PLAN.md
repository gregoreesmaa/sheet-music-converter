# PLAN — `2026-11-26T16-20-ov-32b-gram-weight-high-vicreg`

- **Parent:** `2026-11-26T01-38-ov-32-dinov3-gram-weight-sweep`
- **Provider:** Google Colab `L4` (`BF16`, ~40 min, ~3.2 CU)
- **Arms (500 steps/arm, `small_45m`, VICReg `var=1.0, cov=0.04`):** `gram_weight in (0.5, 1.0, 2.0)`.
- **PROMOTE:** Winning `gram_weight` achieves `dense_erank_mean >= 70.0` AND `dense_offdiag_cosine_mean <= 0.85` AND `global_page_erank >= 15.0` -> lock as default `--gram-weight` in `pretrain_ijepa_rism_v2.py`.
- **DISCARD:** `1.0` and `2.0` underperform `0.5` -> lock `gram_weight=0.5` + VICReg `1.0/0.04`.
