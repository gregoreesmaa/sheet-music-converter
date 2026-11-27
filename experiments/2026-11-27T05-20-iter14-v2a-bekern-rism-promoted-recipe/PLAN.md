# Plan — `2026-11-27T05-20-iter14-v2a-bekern-rism-promoted-recipe`

## 1. Platform Chosen & Rationale

- **Platform:** Local `RTX 4080 SUPER` (`16 GB` VRAM, Ada `cc=8.9`, native `BF16`, CUTLASS `EFFICIENT_ATTENTION`, cuDNN NCHW with `--fixed-hw-buckets`).
- **Rationale:** Zero cloud cost (`$0`), direct local SSD access to `D:\tmp\unlabeled-10k-real\rism_hw`, `D:\tmp\bekern-v2a-cache`, and `D:\tmp\uni_fullpage_iter13_clean.jsonl`, plus full Tier 1–3 throughput (`bs=2, ga=4`, 10-bucket cuDNN NCHW, `BatchedMuon`, `--kd-stop-frac 0.65`).
- **Estimated Wall-Clock:**
  - Phase 0 (`ov-39` RISM pseudo-labels `--limit 1200` + cached `build_iter14_manifest.py --reexport-v2`): `~25–35 min` + `~15 s`
  - Phase 1 (`16,000` full-page Stage-2 steps, warm-started from Iter 13 `ema_latest`): `~4.0–5.0 h`
  - Phase 2 (Post-training evals: `suite31_hw` Fast + `suite31_hw` `+ILMS 0.40/0.85` + 125-page `OMR-NorthStar`): `~25–30 min`
- **Windows WDDM Guardrails:**
  - `$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"`
  - `.\.venv\Scripts\python.exe -u` (unbuffered stdout)
  - `--num-workers 0 --prefetch-threads 6` (zero `.pin_memory()` in threads)
  - Keep peak VRAM `< 14 GB` (avoid silent WDDM system-RAM swap).

---

## 2. Identifiers for Follow-Up Queries

- **Local Run Directories & Logs:**
  - Phase 0 RISM Pseudo-Labels: `D:\tmp\rism-pseudolabels-ov39\uni_rism_pseudolabels_ov39.jsonl`
  - Phase 0 Combined Manifest: `D:\tmp\uni_fullpage_iter14_v2a_rism.jsonl` (stats: `D:\tmp\uni_fullpage_iter14_v2a_rism_stats.json`)
  - Phase 1 `--out-dir`: `D:\tmp\mvd-runs\uni-omr-iter14-v2a-stage2-16k`
  - Combined Log Path *(set on launch)*: `D:\tmp\mvd-runs\uni-omr-iter14-v2a-stage2-16k\launch_iter14.log`
  - Background `taskId`: *(PENDING — fill immediately on launch)*
- **Eval Output JSONs:**
  - Pass 1 (`suite31_hw` Fast `1408px/3584, FSA+MTP-4, mem=4096`): `D:\tmp\uni-omr-suite31-ood\iter14_v2a_stage2_16k_ema_latest_1408_fsa_mtp.json`
  - Pass 2 (`suite31_hw` `+ILMS 0.40/0.85, mem=4096`): `D:\tmp\uni-omr-suite31-ood\iter14_v2a_stage2_16k_ema_latest_1408_fsa_mtp_ilms.json`
  - Pass 3 (`OMR-NorthStar` 125p, `ema_latest`, `--neg-control-gate`): `D:\tmp\northstar-eval-runs\iter14-v2a-last\report.json`
- **Hugging Face (`gregoreesmaa/sheet-music-converter-omr`):**
  - Run Ledger: `runs/2026-11-27T05-20-iter14-v2a-bekern-rism-promoted-recipe.json`
  - Checkpoints: `checkpoints/2026-11-27T05-20-iter14-v2a-bekern-rism-promoted-recipe/`
  - Eval JSON: `evals/2026-11-27T05-20-iter14-v2a-bekern-rism-promoted-recipe_suite31_hw.json`

---

## 3. Required Training & Eval Data (100% Permissive, Zero Eval Overlap)

| Asset | Path | Verified Count / Properties | License & Quarantine Status |
|---|---|---|---|
| **Base `bekern+v2` Phase A Manifest** | `D:\tmp\uni_fullpage_iter14_v2a.jsonl` | `47,727` rows (`26,546` `v2a` re-exported + `20,931` `v1` + `250` `neg_control`), `718,267` new `v2a` tokens across all `28` types, `VOCAB_SIZE=265`, `0` OOV | Permissive only; `0/45` `suite31_hw` overlap, `0/125` `OMR-NorthStar` overlap |
| **Phase 0 `ov-39` RISM Pseudo-Labels** | `D:\tmp\rism-pseudolabels-ov39\uni_rism_pseudolabels_ov39.jsonl` | Generated from `1,200` pages of `D:\tmp\unlabeled-10k-real\rism_hw` (`5,707` total pool) via Iter 13 `ema_latest` (`mean_logprob >= -0.35`, `beat_valid_frac >= 0.70`) | `PDM 1.0`; checked for `0` overlap in `build_iter14_manifest.py` |
| **Phase 0 Combined Manifest** | `D:\tmp\uni_fullpage_iter14_v2a_rism.jsonl` | `47,727` base + kept `ov-39` `real_hw_pseudo` rows; rebuilt in `~15 s` from `D:\tmp\bekern-v2a-cache` | Permissive; ZERO `CC-BY-NC`, ZERO `cvc-muscima`, ZERO `rism-pdm-hw-intl` |
| **Warm-Start Checkpoint** | `D:\tmp\mvd-runs\uni-omr-iter13-v2-stage2-24k\uni_omr_v2_ema_latest.pt` | `Iter 13` `v2` `small_45m` (`vocab_size=237`); `load_v2_checkpoint` copies `0..232` verbatim, remaps `PAD/BOS/EOS/SYS` (`233..236 -> 261..264`), fresh-inits `233..260` | Verified by `tmp-train/mvd/test_iter14_warmstart.py` |
| **Frozen `v1` KD Teacher** | `D:\tmp\mvd-runs\uni-omr-iter10_5-v1\uni_omr_ema_best.pt` | `vocab_size=237`; `forward_and_loss_v2` remaps `261..264 -> 233..236` for teacher input and masks new `v2a` tokens (`233..260`) out of KD; stops & frees at `kd_stop_frac=0.65` | Local verified checkpoint |
| **Unpaired RISM Style Pool** | `D:\tmp\unlabeled-10k-real\rism_hw` | `5,707` archival HW pages for online FDA (`β=0.02`, `p=0.25`) + Color-AdaIN (`p=0.20`) | `PDM 1.0` |
| **Rebuilt MusicXML Exporter** | `converter/assembly-serialize/target/release/assembly-serialize.exe` | Supports `bekern+v2` Phase A `bekern --v2` & `bekern2musicxml` | Local release binary |

---

## 4. Exact Launch Command & Hyperparameters

Use the self-contained PowerShell launcher [launch_iter14.ps1](./launch_iter14.ps1) (supports `-SkipPhase0`, `-SkipTrain`, `-SkipEval`, `-NoCudnn`):

```powershell
powershell -ExecutionPolicy Bypass -File experiments\2026-11-27T05-20-iter14-v2a-bekern-rism-promoted-recipe\launch_iter14.ps1
```

### Phase 0 — `ov-39` RISM Pseudo-Labels + Cached Manifest Rebuild
```powershell
.\.venv\Scripts\python.exe -u tmp-train/mvd/build_rism_pseudolabel_manifest.py `
  --ckpt D:\tmp\mvd-runs\uni-omr-iter13-v2-stage2-24k\uni_omr_v2_ema_latest.pt `
  --input-dir D:\tmp\unlabeled-10k-real\rism_hw `
  --out D:\tmp\rism-pseudolabels-ov39\uni_rism_pseudolabels_ov39.jsonl `
  --device cuda --bf16 --max-dim 1408 --limit 1200 --resume

.\.venv\Scripts\python.exe -u tmp-train/mvd/build_iter14_manifest.py `
  --reexport-v2 `
  --rism-pseudo-jsonl D:\tmp\rism-pseudolabels-ov39\uni_rism_pseudolabels_ov39.jsonl `
  --out D:\tmp\uni_fullpage_iter14_v2a_rism.jsonl
```

### Phase 1 — Iter 14 Stage 2 (`16,000` Steps, `bekern+v2` Phase A + `v3` Strata + `expr_w=2.5`)
```powershell
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
.\.venv\Scripts\python.exe -u tmp-train/mvd/train_uni_v2.py `
  --preset small_45m --max-mem-tokens 3072 `
  --init-from D:\tmp\mvd-runs\uni-omr-iter13-v2-stage2-24k\uni_omr_v2_ema_latest.pt `
  --curriculum fullpage --fullpage-frac 1.00 `
  --fullpage-manifest D:\tmp\uni_fullpage_iter14_v2a_rism.jsonl `
  --max-systems-per-page 16 --stratified-fullpage --stratified-preset v3 `
  --val-count 16 --val-real-hw-count 16 `
  --fullpage-max-dim 1408 --multi-scale --fixed-hw-buckets --max-seq-len 3840 `
  --batch-size 2 --grad-accum 4 `
  --optimizer muon --muon-lr 4.0e-4 --lr 2.5e-4 --weight-decay 0.01 --no-decay-1d --grad-clip 1.0 `
  --lr-schedule wsd --warmup-steps 300 --wsd-decay-frac 0.20 --min-lr-ratio 0.05 --steps 16000 `
  --mixstyle-prob 0.0 --cass-batch-prob 0.5 --cass-prob 0.30 --tok-dropout 0.12 --adaptive-ink-target `
  --augment --aug-prob 0.75 `
  --rism-style-dir D:\tmp\unlabeled-10k-real\rism_hw --rism-fda-prob 0.25 --rism-adain-prob 0.20 --rism-fda-beta 0.02 `
  --teacher-ckpt D:\tmp\mvd-runs\uni-omr-iter10_5-v1\uni_omr_ema_best.pt --teacher-max-dim 1024 `
  --kd-beta 0.5 --kd-alpha 0.30 --kd-temp 1.5 --kd-teacher-ce-cap 1.5 --kd-hw-scale 0.0 --kd-prob 0.5 --kd-stop-frac 0.65 `
  --state-token-loss-weight 2.5 --expr-token-loss-weight 2.5 `
  --qat-int4 --qat-int4-last-frac 0.15 --qat-int4-block-size 64 `
  --gpu-ema-decay 0.9985 --gpu-ema-every 5 `
  --save-ema-milestones 4000 --best-min-step 4000 `
  --eval-every 4000 --eval-max-len 1024 --save-every-minutes 5.0 `
  --bf16 --grad-checkpoint --num-workers 0 --prefetch-threads 6 --seed 42 `
  --out-dir D:\tmp\mvd-runs\uni-omr-iter14-v2a-stage2-16k
```

### Phase 2 — Automatic Post-Training Evals (`ema_latest`)
1. **Pass 1 (`suite31_hw` Fast `FSA+MTP-4`, `--max-mem-tokens 4096`)**
2. **Pass 2 (`suite31_hw` `+ILMS 0.40/0.85`, `--max-mem-tokens 4096`)**
3. **Pass 3 (125-page `OMR-NorthStar` on `ema_latest`, `--neg-control-gate` + rebuilt `assembly-serialize.exe`)**

---

## 5. 5-Minute Checkpointing, Health Check-In & Resume Plan

- **5-Minute Atomic Checkpoints:** `--save-every-minutes 5.0` writes atomic `.tmp -> os.replace` rolling checkpoints (`uni_omr_v2_latest.pt`, `uni_omr_v2_ema_latest.pt`) plus periodic EMA milestones every `4,000` steps (`--save-ema-milestones 4000`) and deferred `best` selection (`--best-min-step 4000`).
- **Mandatory Early Health Check-In (3–5 min after launch):**
  - Confirm Phase 0 RISM pseudo-labels are decoding cleanly (`--bf16`, `mean_logprob`, yield ~20–30%).
  - Once Phase 1 starts, verify `[UniOMRv2 init-from] vocab-expanded tok_emb.weight, lm_head.weight: 237 -> 265`, finite loss, finite `kd_loss`, `nvidia-smi` VRAM `< 14 GB`, GPU util `>= 90%`.
- **Exact Resume Commands (if interrupted):**
  - Resume Phase 0 only (resumable `--resume`): re-run `launch_iter14.ps1`.
  - Skip Phase 0 and run/resume Phase 1 + Phase 2:
    ```powershell
    powershell -ExecutionPolicy Bypass -File experiments\2026-11-27T05-20-iter14-v2a-bekern-rism-promoted-recipe\launch_iter14.ps1 -SkipPhase0 -InitCkpt D:\tmp\mvd-runs\uni-omr-iter14-v2a-stage2-16k\uni_omr_v2_latest.pt
    ```
  - Run Phase 2 Evals only:
    ```powershell
    powershell -ExecutionPolicy Bypass -File experiments\2026-11-27T05-20-iter14-v2a-bekern-rism-promoted-recipe\launch_iter14.ps1 -SkipPhase0 -SkipTrain
    ```

---

## 6. Evaluation Gate & Decision Tree

- **`PROMOTE` (New Production `v2a` Champion):**
  - `OMR-NorthStar` `NS >= 39.0` (up from `34.08`), `S_expr >= 0.65` (up from `0.5544`), `HWM >= 28.0` (up from `23.31`), `NMHR == 0.0%`, AND `suite31_hw` `all45_clean >= 0.715`, `all45_photo >= 0.708`, `decode_ok_rate == 1.0`.
  - **Action:** Promote `uni_omr_v2_ema_latest.pt` as new `v2a` champion, upload to HF, export WebGPU INT4 (`~28 MB`), and run full `learn-from-suite31` + `learn-from-northstar` forensic scouts for Iter 15.
- **`ITERATE` (Partial Win / Slice Asymmetry):**
  - Improves `S_expr` / `OMR-NorthStar` `NS` over Iter 13 (`34.08`) OR improves `suite31_hw` `HWM` / `hw_clean` / `hw_photo`, but misses a gate (e.g. printed regression `> 0.5 pp` or `HWM < 28.0`).
  - **Action:** Activate `learn-from-suite31` + `learn-from-northstar` + `experiment-finished`, diagnose the bottleneck, and immediately queue Iter 14b / Iter 15.
- **`DISCARD` (Regression / Instability):**
  - Regresses on both `OMR-NorthStar` (`NS < 34.0`) and `suite31_hw` (`all45_clean < 0.705` or `all45_photo < 0.695`), or suffers NaN / decode collapse (`decode_ok_rate < 0.95`).
  - **Action:** Document root cause in `OUTCOME.md`, inspect `train_log.json` and `v2a` token gradients, and isolate offending change before re-launching.
