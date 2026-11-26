# `2026-11-26T07-40-i13-04-lejepa-sigreg-vs-vicreg-rism-ssl`

- **Experiment ID:** `2026-11-26T07-40-i13-04-lejepa-sigreg-vs-vicreg-rism-ssl`
- **Short Code:** `i13-04` (Iter 13 Academic Scout 2 — Non-Collapsing Visual SSL)
- **Status:** `QUEUED`
- **Predecessor / Lineage:** Exp 2 (`2026-11-26T00-54-vjepa21-gram-rism-v2`, collapsed `p3_erank=2.1 / mem_erank=2.6`) & Exp 2b (`2026-11-26T02-20-vjepa21-vicreg-anticollapse-rism-v2`, VICReg fix)
- **Academic Basis (2025–2026):**
  - **LeJEPA & SIGReg** (Balestriero & LeCun, Nov 2025, arXiv:2511.08544) & **LeVJEPA** (Aug 2026): proves an isotropic Gaussian embedding minimizes downstream prediction risk and enforces it in linear $O(N \cdot K \cdot M)$ time via **Sketched Isotropic Gaussian Regularization (SIGReg)** using random 1D projections and the **Epps-Pulley characteristic-function test** — avoiding VICReg's noisy $O(D^2)$ covariance estimation at small full-page batch sizes.
  - **$\lambda$-JEPA / SACReg** (Sep 2026): shows backbone dimensional collapse (`p3_erank`) requires direct isotropic regularization on encoder/post-merger tokens rather than projection heads alone.

## Hypothesis (Single Variable)

At our memory-bound full-page SSL batch size (`batch_size=4 × grad_accum=2`), replacing VICReg (`var=1.0, cov=0.04`) on `(mem_post_merger, mem_final)` with **LeJEPA SIGReg** (Epps-Pulley characteristic-function matching across `K=64` random unit slices) will achieve higher `p3_erank` / `mem_erank` (lower `cross_page_cos`, zero dimensional collapse) and stronger downstream `suite31_hw` transfer on `UniOMR v2 small_45m` than VICReg.

## Arms

1. `arm_vicreg`: Exp 2b baseline — V-JEPA 2.1 + KL router + Sobel-HOG + VICReg (`var_w=1.0, cov_w=0.04`).
2. `arm_sigreg_w05`: Identical V-JEPA 2.1 + KL router + Sobel-HOG, replacing VICReg with SIGReg (`weight=0.5`, `K=64` slices, `17` quadrature knots).
3. `arm_sigreg_w15`: Identical V-JEPA 2.1 + KL router + Sobel-HOG, replacing VICReg with SIGReg (`weight=1.5`, `K=64` slices, `17` knots).

Stage 2 audits SVD effective rank & Gaussianity on 12 real RISM pages; Stage 3 runs a matched `600`-step `UniOMR v2 small_45m` supervised transfer probe on `iter10_5_proxy_2k` comparing `arm_vicreg` vs the highest-rank `arm_sigreg_*` arm on `suite31_hw`.
