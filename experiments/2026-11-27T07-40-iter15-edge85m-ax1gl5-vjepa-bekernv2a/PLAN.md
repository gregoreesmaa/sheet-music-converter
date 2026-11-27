# Plan — `2026-11-27T07-40-iter15-edge85m-ax1gl5-vjepa-bekernv2a`

## 1. Platform Chosen & Rationale

- **Platform:** Local `RTX 4080 SUPER` (`16 GB` VRAM, Ada `cc=8.9`, native `BF16`, CUTLASS `EFFICIENT_ATTENTION`, cuDNN NCHW with `--fixed-hw-buckets`; optional `--no-cudnn` fallback via `-NoCudnn`).
- **Rationale:** Zero cloud cost (`$0`), direct local NVMe access to all `11,688` permissive RISM archival pages (`D:\tmp\unlabeled-10k-real\rism_hw` + `D:\tmp\rism-pdm-hw\pages`), newly aligned `1,466`-page gold-XML archival HW pool, expanded `ov-39` RISM pseudo-labels, and Iter 14 `265`-vocab EMA teacher checkpoint.
- **Estimated Wall-Clock:**
  - **Phase 0** (`3,000` Stage-0 V-JEPA+VICReg steps on `edge85m + ax1_gl5`, `bs=2 ga=4`, `max_dim=1024`): `~1.5–2.0 h`
  - **Phase 1** (`4,000` Stage-1 strip warmup steps on `edge85m + ax1_gl5`, `strip_bs=8 ga=4`): `~1.0–1.5 h`
  - **Phase 2** (`20,000` Stage-2 full-page steps on `edge85m + ax1_gl5`, `bs=2 ga=4`, `1408px`, `bekern+v2` Phase A + Iter 14 KD + late INT4 QAT): `~7.0–9.0 h`
  - **Phase 3** (`suite31_hw` Fast + `+ILMS` + INT4-dequant parity + 125-page `OMR-NorthStar` + WebGPU INT4 `< 55 MB` ONNX export check): `~40–55 min`
- **Windows WDDM Guardrails:**
  - `$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"`
  - `.\.venv\Scripts\python.exe -u` (unbuffered stdout)
  - `--num-workers 0 --prefetch-threads 6` (zero `.pin_memory()` inside threads)
  - Keep peak VRAM `< 14 GB` (avoid silent WDDM system-RAM swap).

---

## 2. Identifiers for Follow-Up Queries

- **Local Run Directories & Logs:**
  - Phase 0 Stage-0 V-JEPA `--out-dir`: `D:\tmp\mvd-runs\uni-omr-iter15-edge85m-stage0-vjepa`
  - Phase 1 Stage-1 Strip `--out-dir`: `D:\tmp\mvd-runs\uni-omr-iter15-edge85m-stage1-strip4k`
  - Phase 2 Stage-2 Full-Page `--out-dir`: `D:\tmp\mvd-runs\uni-omr-iter15-edge85m-stage2-20k`
  - Combined Log Path *(set on launch)*: `D:\tmp\mvd-runs\uni-omr-iter15-edge85m-stage2-20k\launch_iter15.log`
  - Background `taskId`: *(PENDING — fill immediately on launch)*
- **Eval & Export Outputs:**
  - Phase 0 Probe JSON: `D:\tmp\mvd-runs\uni-omr-iter15-edge85m-stage0-vjepa\vjepa_iter15_probe.json`
  - Phase 3 Pass 1 (`suite31_hw` Fast `1408px/3584, FSA+MTP-4, mem=4096`): `D:\tmp\uni-omr-suite31-ood\iter15_edge85m_stage2_20k_ema_latest_1408_fsa_mtp.json`
  - Phase 3 Pass 2 (`suite31_hw` `+ILMS 0.40/0.85, mem=4096`): `D:\tmp\uni-omr-suite31-ood\iter15_edge85m_stage2_20k_ema_latest_1408_fsa_mtp_ilms.json`
  - Phase 3 Pass 3 (`suite31_hw` INT4-dequant parity): `D:\tmp\uni-omr-suite31-ood\iter15_edge85m_stage2_20k_ema_latest_int4dequant_1408_fsa_mtp.json`
  - Phase 3 Pass 4 (`OMR-NorthStar` 125p, `ema_latest`, `--neg-control-gate`): `D:\tmp\northstar-eval-runs\iter15-edge85m-last\report.json`
  - Phase 3 Pass 5 (WebGPU INT4 ONNX export `< 55 MB`): `D:\tmp\mvd-runs\uni-omr-iter15-edge85m-stage2-20k\onnx_webgpu_int4`
- **Hugging Face (`gregoreesmaa/sheet-music-converter-omr`):**
  - Run Ledger: `runs/2026-11-27T07-40-iter15-edge85m-ax1gl5-vjepa-bekernv2a.json`
  - Checkpoints: `checkpoints/2026-11-27T07-40-iter15-edge85m-ax1gl5-vjepa-bekernv2a/`
  - Eval JSON: `evals/2026-11-27T07-40-iter15-edge85m-ax1gl5-vjepa-bekernv2a_suite31_hw.json`

---

## 3. Required Training & Eval Data (100% Permissive, Zero Eval Overlap)

| Asset | Path | Verified Count / Properties | License & Quarantine Status |
|---|---|---|---|
| **Unlabeled RISM Archival Pool (A)** | `D:\tmp\unlabeled-10k-real\rism_hw` | `5,707` archival HW pages (Phase 0 SSL + Phase 2 FDA/AdaIN style pool) | `PDM 1.0`; `0` eval overlap |
| **Unlabeled RISM Archival Pool (B)** | `D:\tmp\rism-pdm-hw\pages` | `5,981` extra unique archival HW pages (`11,688` total unique RISM after content-hash dedup) | `PDM 1.0`; `0` eval overlap |
| **Printed SSL Mix Source** | `D:\tmp\uni_fullpage_iter14_v2a.jsonl` | Up to `6,000` printed pages mixed into Phase 0 V-JEPA+VICReg | Permissive; `0/45` `suite31_hw`, `0/125` `OMR-NorthStar` overlap |
| **Stage-1 Printed & HW Strips** | `tmp-train/mvd-sysdata-t32`, `D:\tmp\real-hw-strips-iter7`, `D:\tmp\hw-sim-v06-historical-clean\stage2_strips` | Clean printed + `6,213` real HW + v06 historical sim strips (`0` `mozart-k458`, `0` barline corruption) | Permissive; ZERO `cvc-muscima`, ZERO `v03-clean` |
| **Stage-2 Iter 15 Full-Page Manifest** | `D:\tmp\uni_fullpage_iter15_v2a.jsonl` *(fallback: `D:\tmp\uni_fullpage_iter14_v2a_rism.jsonl`)* | `bekern+v2` Phase A (`vocab_size=265`) + expanded `ov-39` RISM pseudo-labels + newly aligned gold-XML archival HW (`1,466`-page pool) + `neg_control` | Permissive; `0` OOV, `0/45` `suite31_hw`, `0/125` `OMR-NorthStar` overlap |
| **Iter 14 `265`-Vocab EMA Teacher** | `D:\tmp\mvd-runs\uni-omr-iter14-v2a-stage2-16k\uni_omr_v2_ema_latest.pt` | `UniOMRv2Net` `small_45m`, `vocab_size=265` (`bekern+v2` Phase A) for Phase 2 KD | Local verified checkpoint |
| **Rebuilt MusicXML Exporter** | `converter/assembly-serialize/target/release/assembly-serialize.exe` | Supports `bekern+v2` Phase A `bekern --v2` & `bekern2musicxml` | Local release binary |

---

## 4. Exact 4-Phase Launch Commands & Hyperparameters

Use the self-contained PowerShell launcher [launch_iter15.ps1](./launch_iter15.ps1) (supports `-SkipPhase0`, `-SkipPhase1`, `-SkipPhase2`, `-SkipPhase3`, `-NoCudnn`):

```powershell
powershell -ExecutionPolicy Bypass -File experiments\2026-11-27T07-40-iter15-edge85m-ax1gl5-vjepa-bekernv2a\launch_iter15.ps1
```

### Phase 0 — Stage-0 V-JEPA 2.1 + VICReg Encoder Pretraining (`edge85m + ax1_gl5`)
```powershell
.\.venv\Scripts\python.exe -u tmp-train/mvd/pretrain_vjepa_iter15.py `
  --arch-preset edge85m --n-axial-layers 1 --n-global-layers 5 --max-mem-tokens 3072 --adaptive-ink-target `
  --image-dirs D:\tmp\unlabeled-10k-real\rism_hw D:\tmp\rism-pdm-hw\pages `
  --printed-manifest D:\tmp\uni_fullpage_iter14_v2a.jsonl --printed-max-pages 6000 `
  --mask-ratio 0.45 --mask-half-width 0.05 --vicreg-var-weight 1.0 --vicreg-cov-weight 0.04 `
  --disable-random-gram --stroke-hog-weight 0.25 --router-cons-weight 0.2 `
  --optimizer muon --lr 3.0e-4 --weight-decay 0.04 --warmup-steps 150 `
  --steps 3000 --batch-size 2 --grad-accum 4 --max-dim 1024 `
  --bf16 --grad-checkpoint --no-cudnn --prefetch-threads 4 --val-probe-pages 128 `
  --save-every 500 --save-every-minutes 5.0 --seed 42 `
  --out-dir D:\tmp\mvd-runs\uni-omr-iter15-edge85m-stage0-vjepa
```

### Phase 1 — Stage-1 Strip Warmup (`4,000` Steps, `edge85m + ax1_gl5`, `--init-encoder-from` Phase 0)
```powershell
.\.venv\Scripts\python.exe -u tmp-train/mvd/train_uni_v2.py `
  --preset edge85m --n-axial-layers 1 --n-global-layers 5 --max-mem-tokens 3072 --adaptive-ink-target `
  --init-encoder-from D:\tmp\mvd-runs\uni-omr-iter15-edge85m-stage0-vjepa\uni_omr_v2_vjepa_last.pt `
  --curriculum strip --strip-dirs tmp-train/mvd-sysdata-t32 `
  --hw-strip-dirs D:\tmp\real-hw-strips-iter7 D:\tmp\hw-sim-v06-historical-clean\stage2_strips --hw-frac 0.30 `
  --strip-batch-size 8 --strip-grad-accum 4 --grad-accum 4 --strip-height 192 --max-strip-width 1024 --max-seq-len 1024 `
  --optimizer muon --muon-lr 8.0e-4 --lr 2.5e-4 --weight-decay 0.01 --no-decay-1d --grad-clip 1.0 `
  --lr-schedule wsd --warmup-steps 200 --wsd-decay-frac 0.15 --min-lr-ratio 0.05 --steps 4000 `
  --mixstyle-prob 0.0 --cass-prob 0.15 --tok-dropout 0.12 --augment --aug-prob 0.75 `
  --state-token-loss-weight 2.5 --expr-token-loss-weight 2.5 `
  --gpu-ema-decay 0.995 --gpu-ema-every 2 --eval-every 2000 --val-count 32 --save-every-minutes 5.0 `
  --bf16 --grad-checkpoint --no-cudnn --num-workers 0 --prefetch-threads 4 --seed 42 `
  --out-dir D:\tmp\mvd-runs\uni-omr-iter15-edge85m-stage1-strip4k
```

### Phase 2 — Stage-2 Full-Page (`20,000` Steps, `edge85m + ax1_gl5` + `bekern+v2a` + Iter 14 KD + Late INT4 QAT)
```powershell
.\.venv\Scripts\python.exe -u tmp-train/mvd/train_uni_v2.py `
  --preset edge85m --n-axial-layers 1 --n-global-layers 5 --max-mem-tokens 3072 --adaptive-ink-target `
  --init-from D:\tmp\mvd-runs\uni-omr-iter15-edge85m-stage1-strip4k\uni_omr_v2_ema_best.pt `
  --curriculum fullpage --fullpage-frac 1.00 `
  --fullpage-manifest D:\tmp\uni_fullpage_iter15_v2a.jsonl `
  --max-systems-per-page 16 --stratified-fullpage --stratified-preset v3 `
  --val-count 16 --val-real-hw-count 16 `
  --fullpage-max-dim 1408 --multi-scale --fixed-hw-buckets --max-seq-len 3840 `
  --batch-size 2 --grad-accum 4 `
  --optimizer muon --muon-lr 3.5e-4 --lr 2.0e-4 --weight-decay 0.01 --no-decay-1d --grad-clip 1.0 `
  --lr-schedule wsd --warmup-steps 300 --wsd-decay-frac 0.20 --min-lr-ratio 0.05 --steps 20000 `
  --mixstyle-prob 0.0 --cass-batch-prob 0.5 --cass-prob 0.30 --tok-dropout 0.12 `
  --augment --aug-prob 0.75 `
  --rism-style-dir D:\tmp\unlabeled-10k-real\rism_hw --rism-fda-prob 0.25 --rism-adain-prob 0.20 --rism-fda-beta 0.02 `
  --teacher-ckpt D:\tmp\mvd-runs\uni-omr-iter14-v2a-stage2-16k\uni_omr_v2_ema_latest.pt --teacher-max-dim 1024 `
  --kd-beta 0.5 --kd-alpha 0.30 --kd-temp 1.5 --kd-teacher-ce-cap 1.5 --kd-hw-scale 0.0 --kd-prob 0.5 --kd-stop-frac 0.65 `
  --state-token-loss-weight 2.5 --expr-token-loss-weight 2.5 `
  --qat-int4 --qat-int4-start-frac 0.85 --qat-int4-block-size 32 `
  --gpu-ema-decay 0.9985 --gpu-ema-every 5 `
  --save-ema-milestones 4000 --best-min-step 4000 `
  --eval-every 4000 --eval-max-len 1024 --save-every-minutes 5.0 `
  --bf16 --grad-checkpoint --num-workers 0 --prefetch-threads 6 --seed 42 `
  --out-dir D:\tmp\mvd-runs\uni-omr-iter15-edge85m-stage2-20k
```

### Phase 3 — Automatic Post-Training Evals + INT4 Parity + WebGPU `< 55 MB` ONNX Export Check
1. `suite31_hw` Fast (`1408px/3584, FSA+MTP-4, mem=4096`)
2. `suite31_hw` `+ILMS (0.40/0.85, mem=4096)`
3. `suite31_hw` INT4-dequant parity check (round-trip `fake_quant_int4_ste` on `QATInt4ControllerV2` Linear layers, `block_size=32`, compare Note F1 delta `<= 0.5 pp`)
4. 125-page `OMR-NorthStar` (`ema_latest`, `--neg-control-gate` + rebuilt `assembly-serialize.exe`)
5. WebGPU INT4 ONNX export (`export_uni_v2_onnx.py --preset edge85m --targets webgpu --fp16 --nbits-block-size 32`) + automated `< 55 MB` total artifact size check.

---

## 5. 5-Minute Checkpointing, Health Check-In & Resume Plan

- **5-Minute Atomic Checkpoints:** `--save-every-minutes 5.0` on **all three training phases** (`Phase 0`, `Phase 1`, `Phase 2`), plus `--save-ema-milestones 4000` in Phase 2.
- **Mandatory Early Health Check-In (3–5 min after launch of each phase):**
  - **Phase 0:** Confirm `rism=11688` unique pages discovered, `eval_overlap_verified_zero=True`, `erank` healthy (`> 40`), finite VICReg & JEPA loss, VRAM `< 14 GB`.
  - **Phase 1:** Confirm `[UniOMRv2 init-encoder-from]` loaded all `backbone.*` and `encoder.*` keys with `0` shape mismatches, VRAM `< 14 GB`.
  - **Phase 2:** Confirm Iter 14 `265`-vocab EMA teacher KD active, finite `ce_loss`/`kd_loss`, VRAM `< 14 GB`, GPU util `>= 90%`.
- **Exact Resume Commands:**
  - Skip Phase 0 (start at Phase 1): `launch_iter15.ps1 -SkipPhase0`
  - Skip Phases 0–1 and resume Phase 2 from latest rolling ckpt:
    ```powershell
    powershell -ExecutionPolicy Bypass -File experiments\2026-11-27T07-40-iter15-edge85m-ax1gl5-vjepa-bekernv2a\launch_iter15.ps1 -SkipPhase0 -SkipPhase1 -Stage2InitFrom D:\tmp\mvd-runs\uni-omr-iter15-edge85m-stage2-20k\uni_omr_v2_latest.pt
    ```
  - Run Phase 3 Evals + INT4 Parity + ONNX Check only:
    ```powershell
    powershell -ExecutionPolicy Bypass -File experiments\2026-11-27T07-40-iter15-edge85m-ax1gl5-vjepa-bekernv2a\launch_iter15.ps1 -SkipPhase0 -SkipPhase1 -SkipPhase2
    ```

---

## 6. Evaluation Gate & Decision Tree

- **`PROMOTE` (New Production `edge85m` Champion):**
  - Beats Iter 14 on both `suite31_hw` (`all45_clean >= 0.740`, `all45_photo >= 0.730`, `hw_clean >= 0.340`) and `OMR-NorthStar` (`NS >= 45.0`, `HWM >= 34.0`, `NMHR == 0.0%`), AND INT4-dequant parity delta `<= 0.5 pp`, AND WebGPU INT4 ONNX total size `< 55 MB`.
  - **Action:** Promote `uni_omr_v2_ema_latest.pt` as the new production `edge85m` champion, upload weights & eval JSONs to HF, publish WebGPU bundle, and run `learn-from-suite31` + `learn-from-northstar`.
- **`ITERATE` (Partial Win / Slice or INT4 Asymmetry):**
  - Improves `suite31_hw` or `OMR-NorthStar` over Iter 14, OR hits a targeted failure mode (e.g., INT4 parity drop `> 0.5 pp`, WebGPU size `> 55 MB`, or printed vs HW asymmetry).
  - **Action:** Activate `learn-from-suite31`, `learn-from-northstar`, and `experiment-finished`, diagnose root cause, and immediately queue Iter 15b (`depends_on: ["2026-11-27T07-40-iter15-edge85m-ax1gl5-vjepa-bekernv2a"]`).
- **`DISCARD` (Regression / Instability):**
  - Regresses below Iter 14 on both `suite31_hw` and `OMR-NorthStar`, or suffers NaN / decode collapse (`decode_ok_rate < 0.95`).
  - **Action:** Record root cause in `OUTCOME.md` (check Phase 0 probe `R²`, `edge85m` LR stability, and v2-teacher KD loss) before any re-launch.
