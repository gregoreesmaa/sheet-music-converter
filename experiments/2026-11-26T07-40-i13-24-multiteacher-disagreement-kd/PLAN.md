# Plan — `2026-11-26T07-40-i13-24-multiteacher-disagreement-kd`

## 1. Multi-Teacher Ensemble & Disagreement-Gated KD Math

- Both frozen v1 teachers (`T1 = iter10_soup_baseline`, `T2 = iter10_5_best`) and student (`UniOMR v2 small_45m`, `41.3M`) are aligned over their shared append-only vocabulary (`base[:min_base] + 4 specials`).
- At temperature `T = 2.0`, let `P1 = softmax(z1 / T)` and `P2 = softmax(z2 / T)`:
  - Token entropy: `H_m(t) = -sum_v P_m(t, v) log(P_m(t, v) + 1e-8)`.
  - Entropy-inverse weights (`tau_w = 0.5`): `w1(t), w2(t) = softmax([-H1(t)/tau_w, -H2(t)/tau_w])`.
  - Ensemble target: `P_ens(t) = w1(t) * P1(t) + w2(t) * P2(t)`.
  - Teacher disagreement: `dis(t) = JSD(P1(t), P2(t)) in [0, ln 2]`, normalized `dis_norm(t) = dis(t) / ln 2`.
  - Token focus weight: `gamma(t) = 1.0 + lambda_dis * dis_norm(t)` (`lambda_dis = 2.0`), normalized by batch mean so total KD scale stays matched to `arm0`.
  - Objective: Generalized JSD (`beta = 0.5`) between `P_student(t)` and `P_ens(t)` weighted by `gamma(t)` (`alpha = 0.50` KD + `0.50` supervised CE).

## 2. Unlabeled RISM Consensus Span Auxiliary Loss (`arm2`)

- Every 4th optimizer step, draw 1 unlabeled archival page from `datasets/rism_hw_5707.tar` (`max_dim=1280`):
  - Generate a fast greedy draft `y_rism` (`max_len=256`) with `T2` (`iter10_5_ema`, strongest HW teacher).
  - Run teacher-forced forward passes of both `T1` and `T2` on `(I_rism, y_rism)` to obtain `P1, P2`.
  - Construct consensus mask: `M_cons(t) = (argmax(P1(t)) == argmax(P2(t))) & (max(P_ens(t)) >= 0.70)`.
  - If `M_cons.sum() >= 16`, compute `L_rism_kd = G_JSD(P_student, P_ens)` masked to `M_cons` with weight `0.25`.

## 3. Training, Evaluation & Hygiene

- `400` optimizer steps per arm (`batch_size=2`, `grad_accum=4`, `eff_batch=8`, `lr=3e-4` WSD schedule, `BF16`, `seed=42`).
- Evaluate all 3 arms on `proxy_val.jsonl` (`64` pages: `val_ce`, `val_token_f1`) AND `suite31_hw` (`14` HW + `12` stratified Printed = `52` pages, `1280px / max_len=1536`).
- Log wall-clock ratio vs `arm0`, mean teacher disagreement `JSD(T1, T2)`, and `arm2` RISM consensus token yield.
- Read HF token from `/tmp/.hf_token` (purge immediately) and upload `evals/<EXP_ID>.json` and `runs/<EXP_ID>.json`.
