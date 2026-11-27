# `2026-11-27T00-52-i14-04b-adaptive-band-tiling-fixed` — Zero-Retrain Adaptive Band-Tiling Probe (Fixed Harness)

- **Experiment ID:** `2026-11-27T00-52-i14-04b-adaptive-band-tiling-fixed`
- **Short ID / Wave:** `i14-04b` (Wave 11 — Optical Resolution & Token-Cap Wall; `ITERATE` successor to `i14-04`)
- **Status:** `QUEUED`
- **Predecessor:** [`2026-11-26T22-18-i14-04-adaptive-band-tiling-eval`](../2026-11-26T22-18-i14-04-adaptive-band-tiling-eval/OUTCOME.md) (`i14-04`, invalidated by `missing=349` checkpoint loader bug + `[-1, 1]` input norm bug)
- **Platform / GPU:** Colab `1× NVIDIA L4` (`24 GB` VRAM, native BF16; ~15 min, ~1.2 CU; also FP16-safe on Kaggle T4)
- **Model Footprint:** `0` new parameters — evaluates frozen `UniOMR v2 small_45m` checkpoint (`checkpoints/iter12_v2_best/uni_omr_v2_ema_best.pt`)
- **Dataset (100% permissive):**
  - `gregoreesmaa/sheet-music-converter-omr` → `datasets/suite31_hw_eval.tar` (45-piece `suite31_hw` OOD benchmark: 31 printed + 14 real historical HW, `clean` + `photo` = 90 pages)

## Summary & Fixes vs `i14-04`
`i14-04` produced `0.0000` Note F1 across all 3 arms because `_load_frozen_model` missed `ck["model"]`/`ck["config"]` (`missing=349, unexpected=7`), `_pil_to_tensor` fed `[-1, 1]` instead of `[0, 1]`, T4 fell back to FP32, and the autocorrelation `min_lag` latched onto `~3 px` line thickness (marking 85/90 pages dense).

`i14-04b` cleanly re-runs the exact 3-arm zero-retrain probe (`1408px` control vs `1792px` high-res single-pass vs adaptive 2–3 system band-tiling) with all four bugs fixed:
1. **Canonical checkpoint loading:** `eval_uni_suite31.load_uni_checkpoint(ckpt_local, device)` with hard `assert len(meta["missing_keys"]) == 0`.
2. **Canonical `[0, 1]` image preprocessing:** `uni_omr._resize_fullpage_pil` + `uni_omr._pil_to_chw_tensor`.
3. **Explicit AMP dtype:** `torch.bfloat16` on `cc >= 8` (L4/A100), `torch.float16` on `cc < 8` (T4).
4. **Calibrated staff-space estimator:** `min_lag = max(6, int(round(H * 0.0035)))` with local-maximum peak picking (skipping monotonically decaying lag-0 lobe) so sparse pages stay single-pass.
5. **2-page fail-fast sanity gate:** Aborts within `< 30 s` if Arm A fails to decode (`dec_ok == 0` or `note_f1 < 0.05`) on the first 2 printed clean pages.

## Quick Links
- Execution Plan & Decision Tree: [PLAN.md](./PLAN.md)
- Standalone Colab/Modal/Kaggle Runner: [run_colab.py](./run_colab.py)
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-27T00-52-i14-04b-adaptive-band-tiling-fixed.json`
- HF Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-27T00-52-i14-04b-adaptive-band-tiling-fixed.json`
