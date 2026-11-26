# PLAN — `2026-11-26T16-20-ov-31b-vjepa-mask-curriculum-vicreg`

- **Parent:** `2026-11-26T01-38-ov-31-vjepa21-mask-ratio-sweep`
- **Provider:** Google Colab `L4` (`BF16`, ~38 min, ~3.1 CU)
- **Arms (500 steps/arm, `small_45m`, VICReg `var=1.0, cov=0.04`, `gram_weight=0.5`):**
  1. `mask_045_ctrl`: constant `0.45` (`[0.40, 0.50]`)
  2. `mask_065_const`: constant `0.65` (`[0.60, 0.70]`)
  3. `mask_045_to_070_curr`: linear curriculum `0.45 -> 0.70` over steps `1..500`
- **PROMOTE:** Winner beats `mask_045_ctrl` on **both** Ridge Staff `R²` (`>= +0.015`) and kNN-5 Staff `R²` (`>= +0.030`) with `nan_skips == 0`.
- **DISCARD:** Neither beats `mask_045_ctrl` on both probes -> keep constant `(0.45, 0.55)` with VICReg.
