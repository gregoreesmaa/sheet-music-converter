# Plan — `2026-11-26T02-40-iter11-v2-24k` (`UniOMR v2` Iter 11, 24k Steps)

## Hypothesis
Training `UniOMR v2` (`small_45m`, `41.3M` params, ConvNeXt+Axial+2D-RoPE+SwiGLU+GQA+MTP-4+Sobel-HOG+4-edge flood-fill `InkTokenRouterV2`) for `24,000` steps with **hybrid Generalized-JSD distillation from our new record `Iter 10.5` `v1` teacher** (`all45_clean=0.6984`, `all45_photo=0.6832`), `Muon + AdamW` WSD schedule, and last-15% unrotated `INT4 QAT` will surpass `Iter 10.5` on both printed (`>= 0.91`) and real historical handwritten (`>= 0.35`) Strict Note F1 on `suite31_hw` while staying at **`28.04 MB`** WebGPU INT4 (`< 55 MB` deployment gate).

## Exact Launch Configuration
- **Script:** `tmp-train/mvd/train_uni_v2.py`
- **Architecture:** `--preset small_45m --dynamic-routing --max-mem-tokens 10240 --mtp-k 4 --bf16 --channels-last --grad-checkpoint`
- **Distillation:** `--teacher-ckpt D:\tmp\mvd-runs\uni-omr-iter10_5-v1\uni_omr_ema_best.pt --kd-beta 0.5 --kd-alpha 0.30 --kd-temp 1.5`
- **Optimizer & Schedule:** `--optimizer muon --lr 3.0e-4 --lr-schedule wsd --warmup-steps 500 --wsd-decay-frac 0.25 --min-lr-ratio 0.05 --steps 24000`
- **Data:** `--curriculum mixed --fullpage-manifest D:\tmp\uni_fullpage_iter11_clean.jsonl --max-systems-per-page 16 --stratified-fullpage --fullpage-frac 0.80 --strip-dirs tmp-train/mvd-sysdata-t32 --hw-strip-dirs D:\tmp\real-hw-strips-iter7 --hw-frac 0.30 --fullpage-max-dim 1536 --multi-scale --max-seq-len 3840 --augment --aug-prob 0.50 --mixstyle-prob 0.20 --cass-prob 0.10`
- **Losses & QAT:** `--mtp-loss-weight 0.30 --saliency-loss-weight 0.03 --stroke-hog-weight 0.10 --label-smoothing 0.05 --qat-int4 --qat-int4-last-frac 0.15 --qat-int4-block-size 64`
- **Checkpointing & Health:** `--gpu-ema-decay 0.9985 --save-every-minutes 5.0 --eval-every 4000 --val-count 64 --eval-max-len 1536 --batch-size 1 --strip-batch-size 4 --grad-accum 4 --seed 111`

## Early Health Check-In Gate (3–5 min)
- Confirm zero OOMs, VRAM `< 14.0 GB` (WDDM safe), `kd > 0`, `hog` declining, `loss` falling cleanly, throughput stable.

## Promotion Gate (Post-24k `suite31_hw` OOD Eval)
- Beats Iter 10.5 (`all45_clean=0.6984`, `all45_photo=0.6832`) on `suite31_hw` (`45` pieces / `90` pages, `100% DecOK`) with `< 55 MB` WebGPU INT4 export.
