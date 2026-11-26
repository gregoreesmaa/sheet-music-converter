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

## Pre-Handoff Audit (CPU-Verified)
1. **Stage 1 → Stage 2 Handoff (`train_uni_v2.py`):**
   - `uni_omr_v2_ema_best.pt` is atomically written at step `2000` (and updated at `4000` if `ema_ser < best_ema_ser`). Fallback to `uni_omr_v2_ema_latest.pt` only if Stage 1 terminates before step `2000`; also verify whether `ema_best.pt` is step `4000` vs `2000` before launching Stage 2.
   - `--init-from` loads **model weights only** (`load_v2_checkpoint`, `strict=False`); optimizer, WSD step counter (`step=1`), and `GPUEmaShadow` reset cleanly for Stage 2's 20k schedule.
   - Online `v1` KD teacher (`uni-omr-iter10_5-v1/uni_omr_ema_best.pt`, `44.77M`, `backbone_stride=4x8`, `max_dec_len=4096`) passes `1280×960` smoke test cleanly (`n_in=38,400 → n_out=10,244`, `k_kept=10,240`, zero pos-emb/seq-len limit up to `3840`). Monitor peak VRAM `< 14 GB` on dense `1280px` + `T≈3500` batches since frozen FP32 teacher (`~179 MB`) runs forward while student activations are still live.
2. **Manifest Audit (`D:\tmp\uni_fullpage_iter11_clean.jsonl`, `44,734` rows):**
   - `0` missing images, `0` missing bekerns (4,294-row stratified scan), `0` rows with `>16` systems (`max=16, p50=3, p95=7`).
   - Strata (`44,718` train / `16` val at `seed=42`): `legacy_printed` `23,883` (`0.34`, `1.82×`), `multi_renderer_printed` `11,771` (`0.28`, `3.04×`), `synth_hw` `7,771` (`0.26`, `4.28×`), `rare_printed` `824` (`0.06`, `9.32×`), `real_hw` `469` (`0.06`, `16.38×` over `128k` fp draws across `198` unique pieces — healthy with `aug_prob=0.75` + `multi_scale` + `mixstyle`). Note `fp_va` has `0` `real_hw` pages, so `suite31_hw` is mandatory for HW gating.
   - Token lengths: **`0.00% > 3840`** across all 5 strata (`max=3,784` in `synth_hw`, `3,694` in `real_hw`); `3.36% > 2048` (`9.8%` `real_hw`, `7.3%` `synth_hw`) — `--max-seq-len 3840` eliminates 100% of truncation.
3. **Downstream Queue / DAG Update:**
   - `2026-11-26T04-18-academic-gauge-rope-converged-v2` is not yet in `experiments/overnight_queue/queue.json`, and its `PLAN.md`/`README.md`/`run_colab.py` still hard-code `2026-11-26T02-40-iter11-v2-24k` (`checkpoints/iter11_v2_best/uni_omr_v2_ema_best.pt`). Repoint to Iter 12 once Stage 2 completes and uploads to HF.
