# PLAN — `2026-11-26T16-20-ov-33b-vjepa-warmstart-fixed-eval`

- **Parent:** `2026-11-26T01-38-ov-33-vjepa-warmstart-vs-scratch`
- **Provider:** Google Colab `L4` (`BF16`, ~45 min, ~3.6 CU)
- **Fix vs `ov-33`:**
  1. `tmp-train/mvd/eval_sys_gate.py` now uses cross-platform `_BIN_EXT = '.exe' if os.name == 'nt' else ''` so Linux `assembly-serialize` + `eval-harness` actually score all 45 `suite31_hw` pieces.
  2. Evaluates `uni_omr_v2_ema_latest.pt` (step 800, not step-400 `ema_best`) and fails closed if `all45_clean.n != 45` or `note_f1 is None`.
- **PROMOTE:** `arm_b_vjepa_warmstart` beats `arm_a_scratch` by `>= +0.015` on `hw_clean` or `hw_photo` Strict Note F1 with `<= 0.005` regression on `printed_*`.
- **DISCARD:** Warm-start matches or underperforms scratch across both `hw_*` and `printed_*`.
