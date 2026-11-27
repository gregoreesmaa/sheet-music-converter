# `2026-11-27T10-30-i15-11-sheared-strip-conv-and-aniso-pitch-rope2d`

## Hypothesis

Stock `UniOMRv2Net` (`edge85m + ax0_gl6`) has two geometric inductive-bias mismatches on real handwritten and phone-photographed scores:
1. **Axis-aligned `(1, 9)` / `(9, 1)` strips in `AnisotropicMusicBlock`** lose staff-line, beam, and stem coherence under `±4°–8°` camera tilt, page curl, or slanted handwriting.
2. **Isotropic 2D-RoPE (`50%` Y / `50%` X, `base=10000`, `coord_scale=128`)** allocates equal rotary frequency bandwidth to horizontal time and vertical pitch, even though vertical pitch discrimination (staff-line vs space, `~3–5 px` steps) is far more sensitive to fine spatial phase than horizontal spacing.

This 2-arm weight-incompatible proxy tests both upgrades jointly on `edge85m + ax0_gl6` (`600` steps/arm, `iter10_5_proxy_2k`, BF16, Muon+AdamW, seed `42`):
- **Arm 0 (`ctrl`)**: Stock `AnisotropicMusicBlock` (`1x9` / `9x1`) + isotropic `AspectNormalizedRoPE2D`.
- **Arm 1 (`sheared_aniso_rope`)**:
  - Replaces `stage3` and `stage4` `dw_horiz (1, 9)` with **3-angle gated `(3, 9)` sheared strips (`0°, ±5°`)** and `dw_vert (9, 1)` with **3-angle gated `(9, 3)` slanted strips (`0°, ±8°`)**.
  - Replaces `AspectNormalizedRoPE2D` with **Pitch-Biased Anisotropic 2D-RoPE** (`62.5%` Y dims `base_y=2000, coord_scale_y=256` vs `37.5%` X dims `base_x=10000, coord_scale_x=128`).

## Decision Gate

- **PROMOTE `sheared_aniso_rope`** if:
  - Upright `ema_val_ce` matches or beats `ctrl` (`<= ctrl + 0.010` nats, ideally `< ctrl`), AND
  - `±6°` tilted-val CE (`ema_val_tilted_ce`) improves by `>= 0.03` nats over `ctrl`, AND
  - `step_ms` overhead is `<= +8%` and `packed_int4_mb <= 55.0 MB`.
- **ITERATE** if upright `ema_val_ce` improves but tilted CE does not (or vice versa), splitting sheared strips vs anisotropic RoPE2D.
- **DISCARD** if `sheared_aniso_rope` regresses both upright and tilted `ema_val_ce` or adds `> 12%` latency without clear gain.
