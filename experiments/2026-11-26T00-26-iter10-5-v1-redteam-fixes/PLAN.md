# Experiment Plan — `2026-11-26T00-26-iter10-5-v1-redteam-fixes`

## 1. Platform Chosen & Rationale
- **Platform:** Local **RTX 4080 SUPER (16 GB, BF16 native)** — `~$0.05/hr` electricity, fast interactive access to `D:\tmp\` manifests, zero upload latency.
- **Estimated Wall-Clock:** `~90 min` (`8,000` steps @ ~80–85 steps/min on higher-res `1024–1536px` `80%` fullpage batches).

## 2. Identifiers for Follow-Up Queries
- **Local Task ID:** `dd842155-8931-4f97-9016-33cda7cc365a/task-1383`
- **Local Output Dir:** `D:\tmp\mvd-runs\uni-omr-iter10_5-v1`
- **Planned OOD Eval Reports:**
  - `D:\tmp\uni-omr-suite31-ood\iter10_5_step8k_ema_1536_fsa_mtp.json`
  - `D:\tmp\uni-omr-suite31-ood\iter10_5_step8k_ema_1536_clean_gt.json`
- **Hugging Face Sync Target (on completion):** `gregoreesmaa/sheet-music-converter-omr` (`checkpoints/2026-11-26T00-26-iter10-5-v1-redteam-fixes/` and `evals/2026-11-26T00-26-iter10-5-v1-redteam-fixes_suite31_hw.json`).

## 3. Required Training & Eval Data (100% Permissive, 0 Eval Overlap)
- **Full-Page Manifest:** `D:\tmp\uni_fullpage_iter10_5_clean.jsonl` (`44,934` deduplicated pages; stats in `D:\tmp\uni_fullpage_iter10_5_clean_stats.json` & HF `manifests/uni_fullpage_iter10_5_clean_stats.json`).
  - Stratified weights: `legacy_printed 0.34`, `multi_renderer_printed 0.28`, `rare_printed 0.06`, `synth_hw 0.26`, `real_hw 0.06` (`real_hw` strip reuse capped `<= 4×`, `0` dupes, `0` `> 3,840`-tok `<eos>`-poison rows, `0` OSMD banners/percussion stubs).
- **Strip Dirs:** `tmp-train/mvd-sysdata-t32` (printed) + `D:\tmp\real-hw-strips-iter7` (HW, `hw_frac=0.30`).
- **Warm-Start Weights:** `D:\tmp\mvd-runs\uni-omr-iter10-v1\uni_omr_soup.pt`.
- **Eval Manifest:** `data-gen/score-corpus/suite31/suite31_hw_manifest.json` (45 pieces, 90 pages) + companion `suite31_hw_clean_gt_manifest.json`.

## 4. Exact Launch Command (Commit `052e210`)
```powershell
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
.\.venv\Scripts\python.exe -u tmp-train/mvd/uni_omr.py --mode train --curriculum mixed `
  --init-from D:\tmp\mvd-runs\uni-omr-iter10-v1\uni_omr_soup.pt `
  --strip-dirs tmp-train/mvd-sysdata-t32 --hw-strip-dirs D:\tmp\real-hw-strips-iter7 --hw-frac 0.30 `
  --fullpage-manifest D:\tmp\uni_fullpage_iter10_5_clean.jsonl --stratified-fullpage --fullpage-frac 0.80 `
  --strip-height 192 --max-strip-width 1024 --fullpage-max-dim 1536 --multi-scale --max-seq-len 3840 `
  --augment --aug-prob 0.50 --backbone-stride 4x8 --d-model 384 --n-enc-layers 4 --n-dec-layers 6 --n-heads 6 `
  --dynamic-routing --min-keep-ratio 0.06 --max-keep-ratio 0.45 --saliency-thr 0.05 --n-bg-summary 4 `
  --mixstyle-prob 0.20 --tok-dropout 0.0 --cass-prob 0.10 --mtp-k 4 --mtp-loss-weight 0.30 `
  --saliency-loss-weight 0.03 --struct-tok-weight 1.0 --label-smoothing 0.05 `
  --bf16 --channels-last --grad-checkpoint --ckpt-enc-tokens 12000 --ckpt-dec-tokens 4000 `
  --batch-size 1 --strip-batch-size 4 --grad-accum 4 --gpu-ema-decay 0.9985 --ema-only-eval `
  --log-every 10 --lr-schedule wsd --lr 6.0e-5 --warmup-steps 200 --wsd-decay-frac 0.25 --min-lr-ratio 0.05 `
  --steps 8000 --eval-every 4000 --val-count 64 --eval-max-len 1536 --save-every-minutes 5.0 `
  --soup-last-k 0 --seed 47 --device cuda --out-dir D:\tmp\mvd-runs\uni-omr-iter10_5-v1
```

## 5. 5-Minute Checkpointing & Resume Plan
- `--save-every-minutes 5.0` atomically writes both `uni_omr_latest.pt` and `uni_omr_ema_latest.pt` via `.tmp` → `os.replace` in `D:\tmp\mvd-runs\uni-omr-iter10_5-v1\`.
- If interrupted, resume by passing `--init-from D:\tmp\mvd-runs\uni-omr-iter10_5-v1\uni_omr_ema_latest.pt` with remaining steps.

## 6. Evaluation Gate & What We Do With the Result
- **Eval:** Run `eval_uni_suite31.py --max-dim 1536 --max-len 4096 --use-grammar --use-mtp-spec --bf16` on `uni_omr_ema_best.pt` / `uni_omr_ema_latest.pt` (and test `--measure-beam-size 4`).
- **Promote:** If `printed_clean >= 0.905` AND `printed_photo >= 0.900` AND `all45_clean >= 0.70` → promote Iter 10.5 as the new **v1 teacher** for Iter-11 `v1 → v2` hybrid distillation, re-export WebGPU/WASM `web/models/`, and upload checkpoint + eval to HF.
- **Iterate:** If Printed improves but HW dips (or vice versa) → run targeted `learn-from-suite31` forensic audit to adjust `real_hw` weight / `CUT-RISM` mix before Iter 11.
- **Discard:** If it regresses vs Iter 10 on both Printed and HW → keep Iter 10 `soup`/`step12k` as v1 teacher and isolate which of the 15 changes caused the regression via Modal parallel ablations.
