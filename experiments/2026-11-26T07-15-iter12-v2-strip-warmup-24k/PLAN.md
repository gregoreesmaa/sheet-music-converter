# Plan — `2026-11-26T07-15-iter12-v2-strip-warmup-24k`

## Hypothesis
Adding a **4,000-step strip-only warmup** (`bs=32` strips/step) + **decoupled `--muon-lr 1.0e-3 / 8.0e-4`** (`ov-13` + Exp 4) + **`1280px` multi-scale** (`ov-35`) + **un-lagged Stage 1 EMA** (`0.995 / 2`) on top of our `Iter 10.5` hybrid KD teacher will cure Iter 11's slow cold-start convergence (`strip F1 0.45` at 8k) — hitting `strip_val F1 >= 0.85–0.92` inside 45 minutes in Stage 1 and surpassing Iter 10.5 (`0.6984 / 0.6832`) on `suite31_hw` after Stage 2 (`20,000` mixed steps) at `28.04 MB` WebGPU INT4.

## Stage 1 — Strip Warmup (`4,000` steps, ~45 min)
- `--curriculum strip --strip-batch-size 8 --grad-accum 4` (`eff_batch=32` strips/step)
- `--optimizer muon --muon-lr 1.0e-3 --lr 3.0e-4 --lr-schedule wsd --warmup-steps 200 --wsd-decay-frac 0.15 --steps 4000`
- `--gpu-ema-decay 0.995 --gpu-ema-every 2 --eval-every 2000 --save-every-minutes 5.0`
- `--teacher-ckpt D:\tmp\mvd-runs\uni-omr-iter10_5-v1\uni_omr_ema_best.pt --kd-beta 0.5 --kd-alpha 0.30 --kd-temp 1.5`

## Stage 2 — Mixed Full-Page + Unrotated INT4 QAT (`20,000` steps, chained automatically)
- `--init-from D:\tmp\mvd-runs\uni-omr-iter12-v2-stage1-strip4k\uni_omr_v2_ema_best.pt`
- `--curriculum mixed --fullpage-manifest D:\tmp\uni_fullpage_iter11_clean.jsonl --max-systems-per-page 16 --stratified-fullpage --fullpage-frac 0.80 --fullpage-max-dim 1280 --multi-scale --max-seq-len 3840`
- `--optimizer muon --muon-lr 8.0e-4 --lr 3.0e-4 --lr-schedule wsd --warmup-steps 300 --wsd-decay-frac 0.25 --min-lr-ratio 0.05 --steps 20000`
- `--qat-int4 --qat-int4-last-frac 0.15 --qat-int4-block-size 64 --eval-every 4000 --save-every-minutes 5.0`

## Early Health Check-In Gate (4 min)
- Verify VRAM `< 14 GB`, zero OOMs, `kd > 0`, `loss` dropping much faster than Iter 11's cold full-page start, and `strip_val @ 2000` well above Iter 11's `0.453`!
