# Experiment Plan — `2026-11-26T14-30-i13-06b-sar-sam-conf-skip-ttt`

## 1. Motivation (from `i13-06` `OUTCOME.md`)
In `i13-06`, `arm_sar_sam` (`rho=0.01, ent_w=0.5, kl_w=0.5`) delivered **`+0.0378` Note F1 on `hw_photo`** (beating `arm_ctrl_ov38` by `+0.0104` with tighter worst-case losses), but missed `PROMOTE` for two concrete reasons:
1. **Hyperparameter confound:** `arm_sar_sam` ran at `ent_w=0.5` vs `arm_ctrl_ov38`'s `ent_w=1.0`, trailing on `hw_clean` (`+0.0088` vs `+0.0199`).
2. **Unconditional TTT on high-confidence printed pages:** Both `arm_ctrl_ov38` and `arm_sar_sam` failed `pr_photo >= -0.005` due to a single dense printed outlier (`beethoven-grosse-fuge` `-0.0936`). Meanwhile `arm_rhythm_gated` dropped `kl_w=0.0` on failed gates, causing un-anchored drift.

## 2. Experimental Design
- **Base Checkpoint:** `checkpoints/iter10_soup_baseline/uni_omr_soup.pt` (`max_dim=1280, max_len=3840`, MTP-4 + FSA).
- **Matched TTT Base Arms (`lr=3e-5, steps=2, ent_w=1.0, kl_w=0.5, cons_weight=1.0, l2_weight=0.1`):**
  1. `arm_ctrl_ov38` (`sam_rho=0.0`, un-gated control)
  2. `arm_sar_rho005` (`sam_rho=0.005`, un-gated)
  3. `arm_sar_rho010` (`sam_rho=0.010`, un-gated)
- **Confidence Skip Gate Sweep (`mean_logprob > thr` -> 0 TTT updates, exact baseline decode):**
  - Evaluated cleanly for `thr in {-0.08, -0.12, -0.18, -0.25}` on both `arm_sar_rho005` and `arm_sar_rho010` (plus `arm_ctrl_ov38`), leveraging exact mathematical equivalence (skipped pages = baseline; non-skipped pages = un-gated arm) with zero extra GPU cost while reporting exact per-arm `skip_rate`, `hw_clean`, `hw_photo`, `pr_clean`, and `pr_photo` Note F1.
- **Slices:** All `14` HW manuscripts (`28` pages) + `10` printed safety pieces (`20` pages) = `48` pages total.

## 3. Promotion / Discard Gates
- **PROMOTE:** Any `(rho, conf_skip_thr)` arm achieves `hw_clean >= +0.015` AND `hw_photo >= +0.030` AND `pr_clean >= -0.002` AND `pr_photo >= -0.002` (with `beethoven-grosse-fuge` photo untouched / `>= -0.005`).
- **ITERATE:** Clears `pr_photo >= -0.002` and `hw_photo >= +0.025`, but `hw_clean < +0.015` -> tune `ent_w` / `steps=1..3`.
- **DISCARD:** Cannot simultaneously preserve `hw_photo >= +0.020` and `pr_photo >= -0.005` across any `conf_skip_thr`.
