# `2026-11-26T07-40-i13-23-per-measure-process-grpo-curriculum`

- **Batch / Track:** Iter 13 Risky Bets (`batch_08`, Moonshot `i13-23`)
- **Theme:** Token-Aligned Per-Measure Process-Reward GRPO (`PR-GRPO`) + Online Hard-Example Onset-Drift Curriculum
- **Provider / Target:** Google Colab `L4` (`BF16`, `~34 min`, `~2.5 CU`)
- **Base Checkpoint:** `gregoreesmaa/sheet-music-converter-omr:checkpoints/iter10_soup_baseline/uni_omr_soup.pt`
- **Eval Target:** `evals/2026-11-26T07-40-i13-23-per-measure-process-grpo-curriculum.json`

## Motivation & Plateau Addressed

`ov-37` proved that label-free structural RLVR on unlabeled archival RISM pages improves `suite31_hw` HW Note F1 (`+0.015` to `+0.030` on `hw_photo`) with near-zero printed drift (`<= 0.0012`). However, `ov-37` assigns a **single scalar page-level reward** `R_i` uniformly to all 256–512 tokens in a rollout, creating massive credit-assignment noise: valid measures are penalized for an onset drift 4 bars later, and broken measures are rewarded if other bars happen to sum cleanly. Moreover, uniform sampling over the `5,707` RISM pages wastes updates on already-saturated easy pages (`var(R) ≈ 0`) and hopelessly illegible pages (`R ≈ 0`).

## Moonshot Hypothesis

1. **Per-Measure Process-Reward Credit Assignment (`PR-GRPO`):** Splitting each rollout at `<bar>`/`<sys>` boundaries and computing exact **per-measure rhythmic conservation + local no-loop + first-offending-measure blame spike** yields a dense token-aligned advantage `A_{i,t} = 0.65 * A_local(m(t)) + 0.35 * A_page` that surgically repairs bar-onset drift without corrupting valid measures.
2. **Online Hard-Example Onset-Drift Curriculum:** Maintaining a variance-weighted priority buffer over RISM pages (`score = std(R_group) * (1.0 - mean(r_rhythm))`) concentrates GRPO updates on the "learnable hard" frontier where the policy actually has signal to improve.

## Arms (3 Matched Arms × 120 Updates, `G=6`, `max_len=512`, `kl_beta=0.10`, `BF16`, `seed=42`)

1. `arm0_page_grpo_uniform`: Standard page-level scalar GRPO + uniform RISM sampling (`ov-37` baseline).
2. `arm1_measure_pr_grpo_uniform`: Dense token-aligned per-measure `PR-GRPO` + uniform RISM sampling.
3. `arm2_measure_pr_grpo_curriculum`: Dense token-aligned per-measure `PR-GRPO` + online hard-example onset-drift curriculum.

## Kill / Promote Gates (`suite31_hw` 14 HW + 12 Stratified Printed = 52 pages, `1280px / 3840`)

- **PROMOTE:** `arm1` or `arm2` beats `arm0_page_grpo_uniform` by **`>= +0.015` HW avg Note F1** AND **`>= +0.030` HW `r_rhythm`**, with `final_kl <= 0.15` and `printed_clean` drift `<= 0.005`.
- **KILL:** KL explodes (`> 0.35`), non-zero advantage fraction collapses (`< 0.30`), **OR** neither `arm1` nor `arm2` beats `arm0_page_grpo_uniform` by `>= +0.008` HW avg Note F1.
