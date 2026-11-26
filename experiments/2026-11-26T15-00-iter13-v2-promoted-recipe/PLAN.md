# Plan — `2026-11-26T15-00-iter13-v2-promoted-recipe`

## 1. Platform Chosen & Rationale

- **Platform:** Local `RTX 4080 SUPER` (`16 GB` VRAM, Ada `cc=8.9`, native `BF16`, CUTLASS `EFFICIENT_ATTENTION`).
- **Rationale:** Full 2-stage `28,000`-step training (`4k` strip warmup + `24k` fullpage + `15%` INT4 QAT) plus automatic 90-page `suite31_hw` evaluation runs with zero cloud quota burn (`$0`), direct local SSD access to `D:\tmp\` manifests/RISM images, and atomic 5-minute checkpointing.
- **Estimated Wall-Clock:**
  - Stage 1 (`4,000` strip steps, `eff_batch=32`): `~45 min`
  - Stage 2 (`24,000` fullpage steps, `eff_batch=8`, `1280px` multi-scale + FDA/AdaIN): `~6.5–8.0 h`
  - Post-Stage-2 `suite31_hw` Eval (Pass 1 fast + Pass 2 `mbeam=2`): `~15–20 min`
- **Windows WDDM Guardrails:**
  - `$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"`
  - `.\.venv\Scripts\python.exe -u` (unbuffered stdout)
  - `--num-workers 0 --prefetch-threads 4` (no `pin_memory()` in threads)
  - Keep peak VRAM `< 14 GB` (avoid silent WDDM system-RAM swap).

---

## 2. Identifiers for Follow-Up Queries

- **Local Run Directories & Logs:**
  - Stage 1 `--out-dir`: `D:\tmp\mvd-runs\uni-omr-iter13-v2-stage1-strip4k`
  - Stage 2 `--out-dir`: `D:\tmp\mvd-runs\uni-omr-iter13-v2-stage2-24k`
  - Combined Log Path *(set on launch)*: `D:\tmp\mvd-runs\uni-omr-iter13-v2-stage2-24k\launch_iter13.log`
  - Background `taskId`: *(PENDING — fill immediately on launch)*
- **Eval Output JSONs (`D:\tmp\uni-omr-suite31-ood\`):**
  - Pass 1 (Fast `ov-35b` `1408px / 3584` `FSA+MTP`): `D:\tmp\uni-omr-suite31-ood\iter13_v2_stage2_24k_ema_1408_fsa_mtp.json`
  - Pass 2 (High-Acc `ov-34b` + `i13-10` `+ --oemer-dewarp --measure-beam-size 2 --measure-beam-cap 4 --measure-beam-lp 2.0 --measure-beam-max-window 64`): `D:\tmp\uni-omr-suite31-ood\iter13_v2_stage2_24k_ema_1408_fsa_oemer_mbeam2.json`
- **Hugging Face (`gregoreesmaa/sheet-music-converter-omr`):**
  - Run Ledger: `runs/2026-11-26T15-00-iter13-v2-promoted-recipe.json`
  - Checkpoints: `checkpoints/2026-11-26T15-00-iter13-v2-promoted-recipe/`
  - Eval JSON: `evals/2026-11-26T15-00-iter13-v2-promoted-recipe_suite31_hw.json`

---

## 3. Required Training & Eval Data (All Verified on Disk, 100% Permissive)

| Asset | Path | Verified Count / Properties | License & Quarantine Status |
|---|---|---|---|
| **Printed Strip Corpus (Stage 1)** | `tmp-train/mvd-sysdata-t32` | `33,852` verified `(render, bekern)` pairs (`discover_strip_pairs`) | Permissive; `0` eval overlap |
| **Clean HW Strip Corpus (Stage 1)** | `D:\tmp\real-hw-strips-iter7` | `6,213` strips (`35.3%` real HW, `0` barline corruption, `0` `mozart-k458`) | Permissive (`PDM 1.0` / `CC-BY-4.0` / `CC0`); excludes quarantined `hw-sim-v03` |
| **Full-Page Manifest (Stage 2)** | `D:\tmp\uni_fullpage_iter11_clean.jsonl` | `44,734` rows (`0` missing, `0` `>16` systems, `0.00%` `>3840` tokens; `legacy_printed`, `multi_renderer_printed`, `synth_hw`, `rare_printed`, `real_hw` split into archival `10%` + stitched `6%` via `--split-archival-hw`) | Permissive only; **ZERO `CC-BY-NC`**, ZERO `cvc-muscima`, ZERO `rism-pdm-hw-intl` |
| **Unpaired RISM Style Pool (Stage 2 `i13-05`)** | `D:\tmp\unlabeled-10k-real\rism_hw` | `5,707` unlabeled real historical handwritten pages for online FDA (`β=0.02`, `p=0.25`) + Color-AdaIN (`p=0.20`) | `PDM 1.0` (`0` pseudo-label noise; exact synthetic `bekern` GT preserved) |
| **Frozen `v1` KD Teacher** | `D:\tmp\mvd-runs\uni-omr-iter10_5-v1\uni_omr_ema_best.pt` | `Iter 10.5` `v1` EMA (`44.77M`, `all45_clean=0.6984`, `all45_photo=0.6832`). Guarded by `--kd-teacher-ce-cap 1.5 --kd-hw-scale 0.0` so `v1` teacher never poisons HW | Local verified checkpoint |
| **OOD Eval Benchmark** | `suite31_hw_manifest.json` (default in `eval_uni_suite31.py`) | `45` pieces (`31` printed + `14` real HW) × `clean`/`photo` = `90` pages | Canonical benchmark (never legacy 31-piece printed-only) |

> **Note on `ov-39` (`build_rism_pseudolabel_manifest.py`):** Promoted thresholds (`mean_logprob >= -0.35`, `beat_valid_frac >= 0.70` / `0.85`, yielding `~1,484` / `~1,027` pages from the 5,707 RISM pool) are implemented in `tmp-train/mvd/build_rism_pseudolabel_manifest.py` and reserved for the **Iter 14** semi-supervised manifest once the full 5,707-page cloud pseudo-label sweep completes. Iter 13 keeps `uni_fullpage_iter11_clean.jsonl` strictly clean (`0` pseudo-label noise) and leverages RISM-5707 via `i13-05` online FDA + Color-AdaIN pixel transfer.

---

## 4. Exact Launch Commands & Hyperparameters

Use the verified PowerShell launcher [launch_iter13.ps1](./launch_iter13.ps1) (supports `-SkipStage1`, `-SkipStage2`, `-SkipEval`):

```powershell
powershell -ExecutionPolicy Bypass -File experiments\2026-11-26T15-00-iter13-v2-promoted-recipe\launch_iter13.ps1
```

### Stage 1 — Strip Warmup (`4,000` steps, `eff_batch = 32` strips/step)
- **Model & Router (baked at `f089d62`):** `small_45m`, `InkTokenRouterV2` (`p10 + 0.15`, `luma_stop=0.35`, `±6px` border dilation, `vrfm_band_alpha=0.30`, `max_mem_tokens=10240`).
- **Exact CLI Flags (100% verified against `train_uni_v2.py` `argparse`):**
  ```powershell
  $env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  .\.venv\Scripts\python.exe -u tmp-train/mvd/train_uni_v2.py `
    --preset small_45m `
    --curriculum strip `
    --strip-dirs tmp-train/mvd-sysdata-t32 `
    --hw-strip-dirs D:\tmp\real-hw-strips-iter7 `
    --hw-frac 0.30 `
    --strip-batch-size 8 --strip-grad-accum 4 --grad-accum 4 `
    --strip-height 192 --max-strip-width 1024 --max-seq-len 1024 `
    --optimizer muon --muon-lr 1.0e-3 --lr 3.0e-4 --weight-decay 0.01 --no-decay-1d --grad-clip 1.0 `
    --lr-schedule wsd --warmup-steps 200 --wsd-decay-frac 0.15 --min-lr-ratio 0.05 --steps 4000 `
    --mixstyle-prob 0.0 --cass-prob 0.15 --augment --aug-prob 0.75 `
    --teacher-ckpt D:\tmp\mvd-runs\uni-omr-iter10_5-v1\uni_omr_ema_best.pt `
    --kd-beta 0.5 --kd-alpha 0.30 --kd-temp 1.5 --kd-teacher-ce-cap 1.5 --kd-hw-scale 0.0 `
    --gpu-ema-decay 0.995 --gpu-ema-every 2 `
    --eval-every 2000 --val-count 32 --save-every-minutes 5.0 `
    --bf16 --grad-checkpoint --channels-last --num-workers 0 --prefetch-threads 4 --seed 42 `
    --out-dir D:\tmp\mvd-runs\uni-omr-iter13-v2-stage1-strip4k
  ```

### Stage 2 — Pure Full-Page + RISM FDA/AdaIN + Last-15% INT4 QAT (`24,000` steps)
- **Warm-Start:** `--init-from D:\tmp\mvd-runs\uni-omr-iter13-v2-stage1-strip4k\uni_omr_v2_ema_best.pt` (model weights only; optimizer, WSD schedule, and EMA reset cleanly for 24k steps).
- **Exact CLI Flags (100% verified against `train_uni_v2.py` `argparse`):**
  ```powershell
  $env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  .\.venv\Scripts\python.exe -u tmp-train/mvd/train_uni_v2.py `
    --preset small_45m `
    --init-from D:\tmp\mvd-runs\uni-omr-iter13-v2-stage1-strip4k\uni_omr_v2_ema_best.pt `
    --curriculum fullpage --fullpage-frac 1.00 `
    --fullpage-manifest D:\tmp\uni_fullpage_iter11_clean.jsonl `
    --max-systems-per-page 16 --stratified-fullpage --split-archival-hw `
    --val-count 16 --val-real-hw-count 16 `
    --fullpage-max-dim 1280 --multi-scale --max-seq-len 3840 `
    --batch-size 2 --grad-accum 4 `
    --optimizer muon --muon-lr 4.0e-4 --lr 2.5e-4 --weight-decay 0.01 --no-decay-1d --grad-clip 1.0 `
    --lr-schedule wsd --warmup-steps 300 --wsd-decay-frac 0.20 --min-lr-ratio 0.05 --steps 24000 `
    --mixstyle-prob 0.0 --cass-prob 0.15 --augment --aug-prob 0.75 `
    --rism-style-dir D:\tmp\unlabeled-10k-real\rism_hw `
    --rism-fda-prob 0.25 --rism-adain-prob 0.20 --rism-fda-beta 0.02 `
    --teacher-ckpt D:\tmp\mvd-runs\uni-omr-iter10_5-v1\uni_omr_ema_best.pt `
    --kd-beta 0.5 --kd-alpha 0.30 --kd-temp 1.5 --kd-teacher-ce-cap 1.5 --kd-hw-scale 0.0 `
    --qat-int4 --qat-int4-last-frac 0.15 --qat-int4-block-size 64 `
    --gpu-ema-decay 0.9985 --gpu-ema-every 5 `
    --eval-every 4000 --eval-max-len 1024 --save-every-minutes 5.0 `
    --bf16 --grad-checkpoint --channels-last --num-workers 0 --prefetch-threads 4 --seed 42 `
    --out-dir D:\tmp\mvd-runs\uni-omr-iter13-v2-stage2-24k
  ```

### Post-Stage-2 Automatic `suite31_hw` Evaluation
- **Pass 1 (Fast Promoted `ov-35b`):**
  ```powershell
  .\.venv\Scripts\python.exe -u tmp-train/mvd/eval_uni_suite31.py `
    --ckpt D:\tmp\mvd-runs\uni-omr-iter13-v2-stage2-24k\uni_omr_v2_ema_best.pt `
    --max-dim 1408 --max-len 3584 --use-grammar --use-mtp-spec --bf16 `
    --out D:\tmp\uni-omr-suite31-ood\iter13_v2_stage2_24k_ema_1408_fsa_mtp.json
  ```
- **Pass 2 (Promoted High-Accuracy `ov-34b` + `i13-10`):**
  ```powershell
  .\.venv\Scripts\python.exe -u tmp-train/mvd/eval_uni_suite31.py `
    --ckpt D:\tmp\mvd-runs\uni-omr-iter13-v2-stage2-24k\uni_omr_v2_ema_best.pt `
    --max-dim 1408 --max-len 3584 --use-grammar --use-mtp-spec `
    --oemer-dewarp --measure-beam-size 2 --measure-beam-cap 4 --measure-beam-lp 2.0 --measure-beam-max-window 64 --bf16 `
    --out D:\tmp\uni-omr-suite31-ood\iter13_v2_stage2_24k_ema_1408_fsa_oemer_mbeam2.json
  ```
  *(Optional third probe: add `--unitsize-norm` from `i13-10` alongside `--oemer-dewarp` to verify adaptive inter-staffline `max_dim` normalization on `v2`.)*

---

## 5. 5-Minute Checkpointing, Health Check-In & Resume Plan

- **5-Minute Atomic Checkpoints:** Both Stage 1 and Stage 2 pass `--save-every-minutes 5.0` (atomic `.tmp` → `os.replace` for `uni_omr_v2_latest.pt`, `uni_omr_v2_ema_latest.pt`, `uni_omr_v2_best.pt`, and `uni_omr_v2_ema_best.pt`).
- **Mandatory Early Health Check-In (3–5 min after launch):**
  - Check `nvidia-smi` (VRAM `< 14 GB`, GPU util `>= 90%`, BF16 active).
  - Verify `train_log.json` / stdout shows finite `loss`, `kd_loss > 0` on printed batches, `kd_loss == 0` on HW batches (`--kd-hw-scale 0.0`), `strip_val @ 2000` healthy (`SER` dropping sharply), and Stage 2 `fullpage_hw_val` (`16` real HW pages) reporting cleanly at step `4000`.
- **Exact Resume Commands (if interrupted):**
  - Resume Stage 1 from latest checkpoint: re-run Stage 1 command with `--init-from D:\tmp\mvd-runs\uni-omr-iter13-v2-stage1-strip4k\uni_omr_v2_latest.pt` (adjusting `--steps` to remaining steps).
  - Resume Stage 2 only (after Stage 1 already completed):
    ```powershell
    powershell -ExecutionPolicy Bypass -File experiments\2026-11-26T15-00-iter13-v2-promoted-recipe\launch_iter13.ps1 -SkipStage1
    ```
  - Run Post-Stage-2 Eval only:
    ```powershell
    powershell -ExecutionPolicy Bypass -File experiments\2026-11-26T15-00-iter13-v2-promoted-recipe\launch_iter13.ps1 -SkipStage1 -SkipStage2
    ```
- **HF Upload on Completion:** Upload `uni_omr_v2_ema_best.pt` + eval JSONs to `gregoreesmaa/sheet-music-converter-omr` under `checkpoints/2026-11-26T15-00-iter13-v2-promoted-recipe/` and `evals/2026-11-26T15-00-iter13-v2-promoted-recipe_suite31_hw.json`.

---

## 6. Evaluation Gate & Decision Tree

Primary metric: **45-piece `suite31_hw` Strict Note F1 (`90` pages: `31` printed + `14` real HW × `clean`/`photo`)** on `uni_omr_v2_ema_best.pt` (also spot-check `uni_omr_v2_best.pt` raw vs EMA for lag).

- **`PROMOTE` (New Production `v2` Champion + WebGPU INT4 Export):**
  - `all45_clean >= 0.705` AND `all45_photo >= 0.690` (beating Iter 10.5 `v1` `0.6984 / 0.6832`), **with** `hw_clean >= 0.270` and `hw_photo >= 0.225` and `decode_ok_rate == 1.0`.
  - **Action:** Promote as new `v2` champion, upload to HF, export WebGPU INT4 (`~28 MB`) to `web/models/`, run `eval-northstar` (`125` pages), and use as the teacher / pseudo-labeler for **Iter 14** (`ov-39` RISM pseudo-labels + promoted cloud winners).
- **`ITERATE` (Partial Win / Domain Asymmetry / Beats Iter 12 `v2` but Trails Iter 10.5 `v1`):**
  - Beats Iter 12 `v2` (`0.6555 / 0.6114`) but misses Iter 10.5 `v1` on either printed or HW, OR shows `RAW > EMA` lag or QAT degradation (`> 0.01` drop in final 15%).
  - **Action:** Run mandatory `.agents/skills/learn-from-suite31/SKILL.md` (6 parallel CPU eyeball scouts on worst vs best printed & HW), diagnose root cause, and immediately queue targeted Iter-13b / Iter-14 successor.
- **`DISCARD` (Regression / Divergence):**
  - Fails to beat Iter 12 `v2` (`all45_clean < 0.655` or `all45_photo < 0.611`) or suffers NaN / collapse / `decode_ok_rate < 0.95`.
  - **Action:** Document failure mode in `OUTCOME.md`, inspect `train_log.json` for LR / KD / FDA-AdaIN instability, and revert offending knob before re-running.
