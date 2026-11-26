# `2026-11-26T14-30-ov-19b-router-saliency-keep-converged` — Converged `InkTokenRouterV2` `(saliency_thr, max_keep_ratio)` Sweep on Full `suite31_hw`

- **Status:** `PLANNED` (`ITERATE` follow-up to `2026-11-26T01-38-ov-19-router-saliency-keep-sweep`)
- **Platform:** Google Colab `1× NVIDIA L4` (`24 GB`, native `BF16`, `~25–35 min` warm-started / `~90 min` fallback proxy)
- **Baseline Commit SHA:** `36484e2`
- **Model:** `UniOMR v2` (`InkTokenRouterV2`, `dynamic_routing=True`, `min_keep_ratio=0.15`, `max_mem_tokens=3072`, promoted `p10 + 0.15` paper-baseline threshold)
- **Warm-Start / Fallback:** Warm-starts from HF `checkpoints/iter12_v2_best.pt` (or `checkpoints/iter12_v2_best/uni_omr_v2_ema_best.pt`) when available; otherwise trains a `2,500`-step bias-corrected EMA `UniOMR v2 small_45m` proxy on `datasets/iter10_5_proxy_2k.tar`.
- **Eval Dataset:** Full `datasets/suite31_hw_eval.tar` (**`45` pieces = `31` printed + `14` HW × `{clean, photo}` = `90` pages**, `max_dim=1280`, `max_len=3584`, pure-Python MusicXML Strict Note F1, `assert total_scored == 90`).

## Motivation
In `ov-19` (`2026-11-26T01-38-ov-19-router-saliency-keep-sweep`), `(0.08, 0.35)` and `(0.05, 0.45)` breached the `>= 0.96` ink-recall floor on dense pages (`p10 <= 0.76`) while providing zero decode speedup. `arm_003_055` (`0.03, 0.55`) was the best candidate (`gen ink_recall=0.9716`, `2.70×` compression), but `ov-19` ran only `600` scratch steps with `VAL_EVAL_MAX_LEN=512` saturation and un-warmed EMA (`0.997^120 = 0.697` init weight).

## Controlled 3-Arm Sweep (with Promoted `p10 + 0.15` Paper-Baseline Threshold)
1. **`arm_005_075`** (`saliency_thr=0.05`, `max_keep_ratio=0.75`): Current default baseline ceiling.
2. **`arm_002_065`** (`saliency_thr=0.02`, `max_keep_ratio=0.65`): High-recall dense-page safety band.
3. **`arm_003_055`** (`saliency_thr=0.03`, `max_keep_ratio=0.55`): `ov-19` top compression candidate.

Every arm is evaluated on all **`90`** pages of `suite31_hw` (`assert total_scored == 90`, `max_len=3584`) with per-page router telemetry (`ink_recall`, `compression_x`, `keep_frac`) and a dedicated dense-page audit (bottom-10 pages by `ink_recall` where dense piano/orchestral ink could be clipped).

## Quick Links
- Launcher: [run_colab.py](./run_colab.py)
- Plan & Decision Tree: [PLAN.md](./PLAN.md)
- Parent Outcome: [../2026-11-26T01-38-ov-19-router-saliency-keep-sweep/OUTCOME.md](../2026-11-26T01-38-ov-19-router-saliency-keep-sweep/OUTCOME.md)
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-26T14-30-ov-19b-router-saliency-keep-converged.json`
- HF Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-26T14-30-ov-19b-router-saliency-keep-converged.json`
