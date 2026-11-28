# Plan — `2026-11-28T06-45-iter15-pro200m-64k`

> **Pre-Launch Gate (HOLD):** Do **NOT** launch full 64k training until:
> 1. Pending ablations `i14-07`, `i15-04`, and `i15-05` finish and their verdicts are harvested;
> 2. The HF dataset shard audit confirms zero missing/corrupt files, zero eval overlap (`0/45` `suite31_hw`, `0/117` `OMR-NorthStar v0.2`), and 100% permissive licensing;
> 3. A `--smoke-steps 20` GPU verification passes cleanly on Colab.

---

## 1. Platform Chosen & Rationale

- **Training Platform:** Google Colab **H100-80GB** (primary; auto-tunes seamlessly to **A100-40GB** or **L4-24GB** fallback) using native `BF16`, Linux cuDNN `--channels-last` (NHWC on H100/A100) + `--fixed-hw-buckets`, and `BatchedMuon`.
- **Local Evaluation Watcher:** Local `RTX 4080 SUPER` (`16 GB` VRAM, Ada `cc=8.9`, `BF16`, `assembly-serialize.exe`) running [watch_milestones_local.py](./watch_milestones_local.py) asynchronously against HF `checkpoints/iter15_pro200m_64k/milestones/` so milestone scoring never pauses or consumes Colab GPU time.
- **Rationale:** `iter15_pro200m` (`194.14M` infer params) at `1536px` multi-scale + `max_seq_len=3840` + `bd3_repair_head` benefits strongly from H100/A100 VRAM & tensor-core throughput, while 5-minute rolling HF sync + auto-resume makes the 64k run completely resilient to Colab session limits or preemptions.
- **Auto-Tuned GPU Profiles (`run_colab_iter15.py`):**
  - **H100-80GB (`>= 70 GB`):** `bs=4, ga=4` (effective batch `16`), `--grad-checkpoint`, `--channels-last` (Linux NHWC), `num_workers=8`, `prefetch_threads=8`
  - **A100-40GB (`>= 36 GB`):** `bs=4, ga=4` (effective batch `16`), `--grad-checkpoint`, `--channels-last` (Linux NHWC), `num_workers=6`, `prefetch_threads=6`
  - **L4-24GB (`< 36 GB`):** `bs=1, ga=16` (effective batch `16`), `--grad-checkpoint`, `num_workers=2`, `prefetch_threads=4`

---

## 2. All 22+ Promoted Upgrades Wired into `iter15_pro200m`

Verified on disk (`UniOMRv2Config.iter15_pro200m()`, commit `15fc120` / `be914c6`):
- **Total params:** `198,841,845` (`198.84M`) | **Infer params:** `194,136,138` (`194.14M`)
- **Hybrid FP16/INT4 footprint (block-32 + scales):** **`120.41 MB`** (`< 135 MB` target budget)

| # | Category | Promoted Upgrade | Source Exp / Commit | Wired Config / CLI |
|---:|---|---|---|---|
| 1 | Capacity | **`pro200m` (`d_model=768`)** scaling (`12` heads, `8` global enc, `12` dec, `ffn_mult=3.5`) | `i15-40`, `15fc120` | `--preset iter15_pro200m` |
| 2 | Encoder | **All-Global Encoder (`ax0_gl8`)** — drops axial bottleneck for full 2D global attention | `i13-25b`, `15fc120` | `n_axial_layers=0, n_global_layers=8` |
| 3 | Backbone | **Anisotropic InceptionNeXt Music Backbone** (`1x9`, `9x1`, `3x3`, identity split) | Iter 11–14 | `backbone_channels=(64,128,240,448)`, `blocks=(1,2,3,3)` |
| 4 | Router | **Adaptive Per-Page Ink Threshold Router** (`clamp(mean+0.5*std, 0.08, 0.20)`) | `ov-19`/`ov-21` | `--adaptive-ink-target --max-mem-tokens 3072` |
| 5 | Router | **VRFM 3-Row Staff-Band Saliency Smoothing** (`vrfm_band_alpha=0.30`) | Iter 13–14 | Built-in `UniOMRv2Config` default |
| 6 | Attention | **Asymmetric / Higher Cross KV Heads** (`yoco_cross_kv_heads=12`, `yoco_self_kv_heads=6`) | `i13-26b` | `n_kv_heads=6, yoco_cross_kv_heads=12` |
| 7 | Decoder | **YOCO 3-Bank Cross-KV Sharing** (`banks=(0, 3, 7)`) | `i15-36`, `15fc120` | `use_yoco_cla=True, yoco_cross_banks=(0, 3, 7)` |
| 8 | Decoder | **CLA 2:1 Self-Attention KV Sharing** | `i15-36`, `15fc120` | `use_yoco_cla=True, yoco_self_kv_heads=6` |
| 9 | Decoder | **ResFormer Cross-Layer Value Residuals (`v_res_alpha`)** | `i15-36`, `15fc120` | `use_resformer_v_res=True` |
| 10 | Cross-Attn | **Differential Cross-Attention** (`q1/k1`, `lambda_q1`, `subln` noise cancellation) | `i15-33`, `15fc120` | `use_diff_cross_attn=True` |
| 11 | Cross-Attn | **CoordKey-CA Fourier 2D Spatial Key Injection** (`coord_dim=16`, `16` freqs, MLP `128`) | `i15-38`, `15fc120` | `use_coordkey_ca=True` |
| 12 | Decoder | **ModeGate (4-Layer Rank-32 FiLM + Low-Rank Query Gate)** | `i15-37`/`i15-37b` | `mode_gate_layers=4, mode_gate_rank=32` |
| 13 | Decoding | **Chained MTP-4 Speculative Heads** | `i15-37b`, `15fc120` | `mtp_k=4, use_chained_mtp=True` |
| 14 | Aux Head | **Dual-Head Auxiliary Text Head (`lm_head_text`, 129 vocab, stripped at infer)** | `i15-37b`, `15fc120` | `text_aux_vocab_size=129, text_aux_loss_weight=0.02` |
| 15 | Quant | **QuantVLA Per-Head Learnable Attention Temperature (`head_temp`)** | `i15-45`, `15fc120` | `use_quantvla_head_temp=True` |
| 16 | Repair | **BD3-LM + ReMDM Complementary Masked-Diffusion Repair Head** (`2L, d=256, blk=32`) | `i15-46`, `15fc120` | `--bd3-repair-weight 0.25 --bd3-n-views 1` |
| 17 | Data Mix | **`v4_hw70` Stratified Full-Page Mix (`30%` Printed / `69.5%` HW / `0.5%` Neg)** | `i15-09` (`2111604`) | `--stratified-fullpage --stratified-preset v4_hw70` |
| 18 | Vision | **`1536px` Multi-Scale + Fixed H×W Buckets + `max_seq_len=3840`** | `ov-16` / Iter 14 | `--fullpage-max-dim 1536 --multi-scale --fixed-hw-buckets --max-seq-len 3840` |
| 19 | Sim-to-Real | **Online RISM 2D-FFT FDA (`p=0.25, β=0.02`) + Color-AdaIN (`p=0.20`)** | `i13-05` | `--rism-style-dir ... --rism-fda-prob 0.25 --rism-adain-prob 0.20 --rism-fda-beta 0.02` |
| 20 | Optimizer | **Decoupled `Muon + AdamW` + WSD Cooldown + `--no-decay-1d`** *(gates/BD3/CoordMLP/text head routed to AdamW)* | `exp4b`, `ov-13`, `ov-14`, `15fc120` | `--optimizer muon --muon-lr 3.5e-4 --lr 2.0e-4 --lr-schedule wsd --wsd-decay-frac 0.20 --no-decay-1d` |
| 21 | Loss / Reg | **State & Expressive Token Weighting (`2.5×`), Sobel-HOG (`0.15`), CASS (`0.30` after `35%`), Tok-Dropout (`0.12`), Schema Mask v1, MixStyle OFF** | `ov-18`, `ov-24`, `ov-25`, `ov-26` | `--state-token-loss-weight 2.5 --expr-token-loss-weight 2.5 --stroke-hog-weight 0.15 --cass-prob 0.30 --cass-start-frac 0.35 --tok-dropout 0.12 --schema-mask-v1 --mixstyle-prob 0.0` |
| 22 | Quant / EMA / KD | **Late-Stage Hybrid INT4 QAT (`start_frac=0.85, block=32`) + GPU EMA (`0.9985` / every 5) + Exact EMA Milestones + Iter-14 Teacher KD** | `ov-15b`, `ov-41`, `i13-09` | `--qat-int4 --qat-int4-start-frac 0.85 --qat-int4-block-size 32 --gpu-ema-decay 0.9985 --gpu-ema-every 5 --ema-milestone-steps 1000,2000,4000,8000,16000,32000,64000 --keep-ema-milestones 0 --teacher-hf-path checkpoints/iter14_v2a_stage2_16k/uni_omr_v2_ema_latest.pt` |
| 23 | Eval / Post | **FSA Grammar + MTP-4 + OEMER Staff Dewarp + Pillar 3B Neg-Control Gate** | `i13-10`, `ov-34b`, `ov-35b` | Wired identically into `watch_milestones_local.py` (`suite31_hw` + `OMR-NorthStar v0.2`, `--no-cudnn` on Windows) |

---

## 3. Identifiers for Follow-Up Queries

- **Colab Session Name:** `iter15-pro200m-64k` *(TBD on launch)*
- **Colab Local Paths:**
  - Data root: `/content/data`
  - Run root: `/content/runs/iter15_pro200m_64k`
- **Hugging Face Private Hub (`gregoreesmaa/sheet-music-converter-omr`):**
  - Rolling Latest Checkpoints (5-min sync): `checkpoints/iter15_pro200m_64k/latest/` (`uni_omr_v2_latest.pt`, `uni_omr_v2_ema_latest.pt`, `train_log.json`, `run_state.json`)
  - Permanent EMA Milestones: `checkpoints/iter15_pro200m_64k/milestones/step_001000_ema.pt`, `step_002000_ema.pt`, `step_004000_ema.pt`, `step_008000_ema.pt`, `step_016000_ema.pt`, `step_032000_ema.pt`, `step_064000_ema.pt`
  - KD Teacher Checkpoint: `checkpoints/iter14_v2a_stage2_16k/uni_omr_v2_ema_latest.pt`
  - Run Ledger: `runs/2026-11-28T06-45-iter15-pro200m-64k.json`
- **Local Milestone Watcher Paths (Windows RTX 4080 SUPER):**
  - Downloaded milestones & eval artifacts: `D:\tmp\iter15-milestones\`
  - `suite31_hw` manifest (90p): `data-gen/score-corpus/suite31/suite31_hw_manifest.json`
  - `OMR-NorthStar v0.2` manifest (117p): `D:\tmp\northstar-eval-pack-v0.2\manifest.json`
  - Live leaderboard JSON: `experiments/2026-11-28T06-45-iter15-pro200m-64k/milestones.json`

---

## 4. Required Training & Eval Data (100% Permissive, Zero Eval Overlap)

- **HF Dataset Prefix:** `datasets/iter15_v2a/` (full-page shards + `uni_fullpage_iter15_v2a.jsonl`, remapped cleanly from `D:\tmp\...` -> `/content/data/...`)
- **Core HF Shards:** `datasets/rism_hw_5707.tar` (unpaired RISM style pool for FDA/AdaIN)
- **Stratified Weights (`v4_hw70`):**
  - Printed (`30.0%`): `legacy_printed 13.2%`, `multi_renderer_printed 13.2%`, `rare_printed 3.6%`
  - Handwritten (`69.5%`): `synth_hw 25.0%`, `real_hw_archival 18.0%`, `real_hw_pseudo 20.0%`, `real_hw_stitched 6.5%`
  - Negative Controls (`0.5%`): `neg_control 0.5%`
- **Quarantine Guarantee:** ZERO `cvc-muscima` (`CC-BY-NC`), ZERO `hw-sim-train-v03-clean`, ZERO `rism-pdm-hw-intl`, ZERO overlap with `suite31_hw` (`45` pieces) or `OMR-NorthStar v0.2` (`117` pages).

---

## 5. Exact Launch, Smoke-Test, Resume & Watcher Commands

### A. CPU Dry-Run Self-Test (Local Verification)
```powershell
.\.venv\Scripts\python.exe experiments\2026-11-28T06-45-iter15-pro200m-64k\run_colab_iter15.py --dry-run
```

### B. Pre-Launch GPU Smoke Test (`20` steps on Colab)
```bash
python3 -u experiments/2026-11-28T06-45-iter15-pro200m-64k/run_colab_iter15.py --smoke-steps 20
```

### C. Full 64k Training Run on Colab (Auto-Resumes Automatically from HF)
```bash
python3 -u experiments/2026-11-28T06-45-iter15-pro200m-64k/run_colab_iter15.py --steps 64000
```

### D. Local Milestone Watcher (RTX 4080 SUPER — Evaluates `1k, 2k, 4k, 8k, 16k, 32k, 64k`)
```powershell
.\.venv\Scripts\python.exe -u experiments\2026-11-28T06-45-iter15-pro200m-64k\watch_milestones_local.py
```
*(Pass `--once` to evaluate currently available milestones and exit; `--poll-seconds 120` is the continuous daemon default.)*

---

## 6. 5-Minute Checkpointing, Permanent Milestones & Auto-Resume Plan

1. **Local Atomic Saves:** `train_uni_v2.py` runs with `--save-every-minutes 5.0` (atomic `.tmp -> os.replace` for `uni_omr_v2_latest.pt` + `uni_omr_v2_ema_latest.pt`) and `--ema-milestone-steps 1000,2000,4000,8000,16000,32000,64000 --keep-ema-milestones 0` (saving exact `uni_omr_v2_ema_step{m:06d}.pt` files at the 7 log-spaced milestone steps without pruning).
2. **Background 5-Min HF Syncer (`HFCheckpointAndMilestoneSyncer`):**
   - Copies changed files to an atomic `.sync_stage/*.tmp` snapshot inside `try/except OSError` before upload (never races an active save) with 5× exponential backoff retries.
   - Uploads rolling `uni_omr_v2_latest.pt`, `uni_omr_v2_ema_latest.pt`, `train_log.json`, and `run_state.json` to `checkpoints/iter15_pro200m_64k/latest/` every 5 minutes and on `finally` flush.
   - Archives **only exact** `uni_omr_v2_ema_step{m:06d}.pt` files for **`1000, 2000, 4000, 8000, 16000, 32000, 64000`** once to `checkpoints/iter15_pro200m_64k/milestones/step_{m:06d}_ema.pt` (never backfills milestones from `uni_omr_v2_ema_latest.pt`).
3. **Zero-Loss True `--resume` Auto-Resume:**
   - On startup, `run_colab_iter15.py` checks `checkpoints/iter15_pro200m_64k/latest/` on HF, restores `run_state.json` + `uni_omr_v2_latest.pt` + `uni_omr_v2_ema_latest.pt`, and launches `train_uni_v2.py --resume uni_omr_v2_latest.pt --steps 64000` (keeping `--steps 64000` fixed!).
   - Because true `--resume` restores the exact global step, Muon/AdamW optimizer state, GPU EMA weights, WSD cooldown schedule, CASS ramp, and late INT4 QAT schedule seamlessly across Colab preemptions, no `step_offset` hack is needed.

---

## 7. Evaluation Gate & Decision Tree

Every permanent EMA milestone (`1k, 2k, 4k, 8k, 16k, 32k, 64k`) is scored automatically on:
- **`suite31_hw` (90 pages):** `printed_clean`, `printed_photo`, `hw_clean`, `hw_photo`, `all45_clean`, `all45_photo`, `decode_ok_rate`
- **`OMR-NorthStar v0.2` (117 pages):** `NS Score`, `Tier-W NS`, `CDR / ALS / RWP / HWM`, `S_note / S_struct / S_expr / S_play`, `median_wFPP`, `NMHR`

Decision tree at 64k (and early-stop check at 8k/16k/32k):
- **`PROMOTE` (New Production Flagship `pro200m` Champion):**
  - `suite31_hw`: `all45_clean >= 0.780`, `all45_photo >= 0.760`, `hw_clean >= 0.450`, `printed_clean >= 0.900`, `decode_ok_rate >= 0.98`
  - `OMR-NorthStar v0.2`: `NS >= 55.0`, `HWM >= 42.0`, `NMHR == 0.0%`
  - Hybrid INT4 size `< 135 MB` (`120.41 MB` verified) and INT4-dequant parity delta `<= 0.5 pp`.
  - **Action:** Promote winning EMA milestone to production, run `learn-from-suite31` + `learn-from-northstar`, and export quantized inference bundles.
- **`ITERATE` (Clear Scaling Gain with Remaining Slice/Domain Gap):**
  - Beats Iter 14 / Iter 15-edge85m on `suite31_hw` or `OMR-NorthStar v0.2`, or reveals a targeted bottleneck (e.g. printed regression under `v4_hw70`, BD3 repair head under-utilization, or late-QAT drop).
  - **Action:** Trigger `learn-from-suite31` + `learn-from-northstar` + `experiment-finished`, diagnose root cause, and immediately queue `iter15b-pro200m` (`depends_on: ["2026-11-28T06-45-iter15-pro200m-64k"]`).
- **`DISCARD` (Instability / Scaling Failure):**
  - NaN/divergence, decode collapse (`decode_ok_rate < 0.95`), or flat/regressed curve at `8k -> 16k` below Iter 14.
  - **Action:** Stop run early to conserve Colab CU, record forensic root cause in `OUTCOME.md`, and isolate offending upgrade before re-launch.
