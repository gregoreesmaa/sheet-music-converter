# PLAN — `2026-11-26T16-20-i13-06c-sar-sam-tight-conf-skip`

- **Parent:** `2026-11-26T14-30-i13-06b-sar-sam-conf-skip-ttt`
- **Provider:** Google Colab `L4` (`BF16`, ~32 min, ~2.6 CU)
- **Base Arms (`lr=3e-5, kl_w=0.5`):**
  1. `arm_ctrl_ent05_s2` (`sam_rho=0.0, ent_w=0.5, steps=2`)
  2. `arm_sar_rho010_ent05_s2` (`sam_rho=0.01, ent_w=0.5, steps=2`)
  3. `arm_sar_rho010_ent075_s2` (`sam_rho=0.01, ent_w=0.75, steps=2`)
  4. `arm_sar_rho010_ent05_s3` (`sam_rho=0.01, ent_w=0.5, steps=3`)
- **Tight Confidence Skip Thresholds:** `(-0.035, -0.040, -0.045)`.
- **PROMOTE:** Any `(arm, thr)` achieves `hw_clean >= +0.015` AND `hw_photo >= +0.030` AND `pr_clean >= -0.002` AND `pr_photo >= -0.002` (with `beethoven-grosse-fuge` untouched).
- **DISCARD:** No arm hits `hw_photo >= +0.025` + `pr_photo >= -0.002`.
