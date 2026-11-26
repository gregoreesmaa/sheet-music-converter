# `2026-11-26T07-40-i13-06-memo-sar-rhythm-gated-ttt`

- **Experiment ID:** `2026-11-26T07-40-i13-06-memo-sar-rhythm-gated-ttt`
- **Short Code:** `i13-06` (Iter 13 Academic Scout 2 — Reliability-Gated & Multi-View Test-Time Adaptation)
- **Status:** `QUEUED`
- **Predecessor / Lineage:** `ov-38` (`2026-11-26T01-38-ov-38-scribe-ttt-lr-steps-sweep`), `ov-38b` (`2026-11-26T06-25-ov-38b-scribe-ttt-budget-cons-3840`), and `ov-39` (`2026-11-26T01-38-ov-39-rism-pseudolabel-threshold-sweep`)
- **Academic Basis (2024–2026):**
  - **MEMO** (Test-Time Robustness via Marginal Entropy Minimization over Augmented Views, adapted to HTR/document TTA 2024–2026): minimizes entropy of the *marginal* distribution $\bar{p} = \frac{1}{K}\sum_k p(y|a_k(x))$ across views plus Jensen-Shannon consistency, reducing overconfident single-view drift.
  - **SAR** (Sharpness-Aware & Reliable Entropy Minimization, Niu et al.; 2024–2026 sequence TTA): prevents TTA collapse under distribution shift by taking a 1-step sharpness-aware (SAM, $\rho=0.01$) perturbation toward flat minima and filtering unreliable high-entropy updates.
  - **DocTTT** (Gu et al., WACV 2025) & **OV-39 Structural Reliability Gating**: `ov-38` proved per-token `conf >= 0.85` fires on `82–94%` of tokens even on `0.044` F1 pages (zero filtering), whereas `ov-39` proved sequence-level `mean_logprob >= -0.35` + `beat_valid_frac >= 0.70` + `r_noloop == 1.0` accurately separates reliable from hallucinating manuscripts.

## Hypothesis (Single Variable)

Replacing `ov-38`'s single-view entropy update (`ent_weight=1.0`) inside `full_encoder` TTT (`lr=3e-5, steps=2`, canonical `max_dim=1280, max_len=3840`) with **MEMO marginal multi-view entropy**, **SAR sharpness-aware perturbation**, or **OV-39 rhythm-reliability gating** (enabling decoder entropy only when `mean_logprob >= -0.35`, `beat_valid_frac >= 0.70`, `r_noloop == 1.0`, else pure visual multi-view consistency) will eliminate regressions on hard manuscripts (`hw-debussy-lindaraja`, `hw-mozart-k387`) and cross the `+0.020` `suite31_hw` PROMOTE gate with zero printed regression.

## Arms (4 Matched Arms at Canonical `1280px / 3840`)

1. `arm_ctrl_ov38`: `ov-38` winning baseline (`full_encoder`, `lr=3e-5`, `steps=2`, `ent=1.0, kl=0.5`, single-view entropy).
2. `arm_memo_marginal`: MEMO marginal entropy $H(\bar{p}) + 0.5\cdot\text{JSD}(p_{\text{clean}}, p_{\text{aug}})$ across clean + jittered views.
3. `arm_sar_sam`: SAR sharpness-aware 1-step SAM perturbation ($\rho=0.01$) on `full_encoder` around the `ov-38` objective.
4. `arm_rhythm_gated`: OV-39 structural-reliability gate — if pre-TTT pseudo-decode passes `mean_logprob >= -0.35`, `beat_valid_frac >= 0.70`, and `r_noloop == 1.0`, run `cons + 0.5*ent + 0.5*kl`; otherwise run pure visual consistency (`ent=0.0, kl=0.0`).
