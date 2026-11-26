# 2026-11-26T16-20-ov-32b-gram-weight-high-vicreg

Follow-up to `2026-11-26T01-38-ov-32-dinov3-gram-weight-sweep` (`ITERATE`).
Sweeps `gram_weight in (0.5, 1.0, 2.0)` with **VICReg (`var_w=1.0, cov_w=0.04`)** enabled to eliminate the `0.993` DC-mode anisotropy observed in `ov-32` while checking if higher `gram_weight` continues lifting dense `erank`.
