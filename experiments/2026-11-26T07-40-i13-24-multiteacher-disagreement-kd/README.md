# `2026-11-26T07-40-i13-24-multiteacher-disagreement-kd`

- **Batch / Track:** Iter 13 Risky Bets (`batch_08`, Moonshot `i13-24`)
- **Theme:** Disagreement-Gated Multi-Teacher Ensemble KD + Unlabeled RISM Consensus Span Distillation (`MT-DKD`)
- **Provider / Target:** Google Colab `L4` (`BF16`, `~36 min`, `~2.8 CU`)
- **Student:** `UniOMR v2 small_45m` (`41.3M`, random init, `max_dim=1280`, `max_seq_len=1536`)
- **Teachers (HF Verified):**
  - `T1`: `checkpoints/iter10_soup_baseline/uni_omr_soup.pt` (strong printed + soup smoothing)
  - `T2`: `checkpoints/iter10_5_best/uni_omr_ema_best.pt` (all-time `hw_photo` record `+6.58 pp` over `T1`)
- **Eval Target:** `evals/2026-11-26T07-40-i13-24-multiteacher-disagreement-kd.json`

## Motivation & Plateau Addressed

Single-teacher online distillation (`ov-42`) transfers every blind spot of `uni_omr_soup.pt` (e.g., `hw_photo` `0.1516` vs `Iter 10.5` `0.2174`) and weights all tokens uniformly — spending most KD capacity on trivial headers/barlines instead of the hard polyphonic and handwritten passages where `UniOMR v2` actually struggles during cold-start (`Iter 11` stopped early at 8k steps).

## Moonshot Hypothesis

1. **Entropy-Inverse Multi-Teacher Ensemble (`T1 + T2`):** Weighting each teacher per token by `w_m(t) ∝ exp(-H(P_m(t)) / tau_w)` lets the sharper teacher dominate locally (e.g., `T2` on archival HW photo ink, `T1` where soup is better calibrated).
2. **Disagreement-Gated Token Focus:** Upweighting the KD loss by `gamma(t) = 1 + lambda_dis * JSD(P1(t), P2(t))` concentrates student gradient on the exact hard tokens where the two v1 champions diverge.
3. **Unsupervised RISM Consensus Span Distillation:** Every 4th step, feeding an unlabeled `rism_hw` page through `T1` and `T2` and distilling only on token spans where `argmax(P1) == argmax(P2)` AND `max(P_ens) >= 0.70` injects high-precision archival ink supervision directly into `UniOMR v2` from step 1!

## Arms (3 Matched Arms × 400 Steps, `iter10_5_proxy_2k`, `1280px`, `BF16`, `seed=42`)

1. `arm0_single_teacher_kd`: Single-teacher `T1` G-JSD KD (`alpha=0.50`, `T=2.0`) — `ov-42` baseline.
2. `arm1_multi_teacher_ent_disagree_kd`: `T1 + T2` entropy-weighted ensemble + JSD disagreement-gated token weighting (`lambda_dis=2.0`).
3. `arm2_mt_kd_plus_rism_consensus`: `arm1` + every-4th-step unlabeled `rism_hw` multi-teacher consensus span distillation.

## Kill / Promote Gates (`proxy_val` 64p + `suite31_hw` 14 HW + 12 Printed = 52p, `1280px`)

- **PROMOTE:** `arm1` or `arm2` beats `arm0_single_teacher_kd` by **`>= +0.025` `proxy_val` Token F1** (or **`>= -0.05` `val_ce`** / **`>= +0.015` `suite31_hw` HW Note F1**) with wall-clock overhead `<= 1.45×` `arm0`.
- **KILL:** Wall-clock overhead `> 1.75×` **OR** neither `arm1` nor `arm2` beats `arm0` on `val_ce` and `val_token_f1` (disagreement amplifies label noise instead of frontier signal).
