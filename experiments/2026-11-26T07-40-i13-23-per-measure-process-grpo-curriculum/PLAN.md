# Plan — `2026-11-26T07-40-i13-23-per-measure-process-grpo-curriculum`

## 1. Per-Measure Process-Reward Formulation (`compute_token_process_rewards`)

Given a sampled rollout `y_i = [t_1, ..., t_L]` and active time signature tracked from `time-N-D` tokens:
1. Partition `y_i` into measure spans `M_1, ..., M_K` delimited by `<bar>` and `<sys>` tokens (header tokens before the first note belong to span `M_0`).
2. For each measure `M_k` (`k >= 1`):
   - Compute per-voice summed durations (`dur-*`, `dot`, tuplets) using the exact duration table from `rlvr_grpo.py`.
   - `r_bar(M_k) = 1.0` if every active voice in `M_k` satisfies `|sum_dur - bar_len| <= 1e-3` (with at least 1 note/rest), else `max(0.0, 1.0 - min(1.0, err / bar_len))`.
   - `r_local_loop(M_k) = 0.0` if `M_k` participates in a repeated n-gram loop (`>= 3x`), else `1.0`.
   - **First-Offending-Measure Blame:** Identify the first measure `k*` with `r_bar(M_{k*}) < 0.5`; subtract `-0.35` from `M_{k*}` and apply mild decay (`0.85^(k - k*)`) to downstream measures so the policy gradient focuses squarely on the root onset-drift bar.
3. Combine with page-level structural reward `R_page`:
   - `r_tok(i, t) = 0.65 * r_bar_shaped(m(t)) + 0.35 * R_page(i)`.
   - Normalize across the `G=6` rollouts at each token position / rollout group: `A_{i,t} = (r_tok(i, t) - mean_G(r_tok)) / (std_G(r_tok) + 1e-6)` clamped to `[-4.0, 4.0]`.
   - Compute clipped GRPO surrogate + exact `k3` KL penalty (`kl_beta=0.10`) against frozen `ref_model`.

## 2. Online Hard-Example Onset-Drift Curriculum (`arm2`)

- Maintain a priority dictionary over a 600-page active working pool drawn from `datasets/rism_hw_5707.tar`:
  - First 30 updates: exploration pass sampling unseen pages uniformly, recording `priority(p) = std(R_group) * (1.0 - mean(r_rhythm)) + 0.05`.
  - Updates 31–120: 70% softmax-priority sampling from scored hard-frontier pages (temperature `0.15`) + 30% fresh exploration from unseen RISM pages to prevent overfitting to a tiny subset.

## 3. Matched Evaluation & Telemetry

- Evaluate baseline + all 3 arms on `suite31_hw` (`14` HW + `12` stratified Printed = `52` pages) at canonical `max_dim=1280`, `eval_max_len=3840` using `generate_mtp_spec` + FSA grammar + `rlvr_grpo.compute_sequence_f1`.
- Log per-arm `final_kl`, `max_kl`, `nonzero_adv_frac`, `mean_r_rhythm`, `curriculum_unique_pages`, and per-piece F1 deltas.
- Read token from `/tmp/.hf_token` (purge immediately) and upload `evals/<EXP_ID>.json` and `runs/<EXP_ID>.json` to `gregoreesmaa/sheet-music-converter-omr`.
