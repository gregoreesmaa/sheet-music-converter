# PLAN — `2026-11-27T08-19-i15-09-data-mix-ratio-and-rep-cap-sweep`

## 1. Platform Chosen & Rationale

- **Platform:** Google Colab `L4` / `H100` (`provider: "colab_l4"`, `gpu: "L4"`, Wave 13, `priority: 9`, BF16 native).
- **Rationale:** 3 controlled warm-start fine-tuning arms (`800` steps/arm, `grad_accum=4`, `3,200` micro-batches/arm) + full 90-page `suite31_hw` RAW & EMA evaluation per arm. Warm-starting from the frozen `UniOMRv2` EMA checkpoint makes 800 steps sufficient to measure clean Printed/HW drift and archival overfitting sensitivity while staying well within a single Colab session.
- **Estimated Wall-Clock & Quota:** **~45 min on Colab L4** / **~10 min on Colab H100** (~**`3.5` CU**).

## 2. Identifiers for Follow-Up Queries

- **Queue Entry:** `experiments/overnight_queue/queue.json` (`id: "2026-11-27T08-19-i15-09-data-mix-ratio-and-rep-cap-sweep"`, `short_id: "i15-09"`, `wave: 13`, `priority: 9`, `provider: "colab_l4"`, `gpu: "L4"`, `status: "QUEUED"`).
- **Hugging Face (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-27T08-19-i15-09-data-mix-ratio-and-rep-cap-sweep.json`
  - Run ledger: `runs/2026-11-27T08-19-i15-09-data-mix-ratio-and-rep-cap-sweep.json`
- **Live Cloud IDs (filled at launch):**
  - Colab session / notebook ID: `TBD`

## 3. Required Training & Eval Data

- **Warm-Start Checkpoint:** Frozen `UniOMRv2` EMA loaded via canonical `eval_uni_suite31.load_uni_checkpoint` (asserts `missing_keys == 0`), resolved from HF candidate chain:
  1. `checkpoints/iter14_v2a_best/uni_omr_v2_ema_best.pt`
  2. `checkpoints/iter13_v2_best/uni_omr_v2_ema_best.pt`
  3. `checkpoints/iter12_v2_best/uni_omr_v2_ema_best.pt`
  4. `checkpoints/iter11_v2_best/uni_omr_v2_ema_best.pt`
  5. `checkpoints/iter10_5_v2_best/uni_omr_v2_ema_best.pt`
- **Training Pool:** `datasets/iter10_5_proxy_2k.tar` (plus optional `datasets/rism_pseudolabels_ov39.tar` if present on HF; permissive only, **ZERO `CC-BY-NC`**, `0 / 45` `suite31_hw` overlap). If `neg_control` rows are absent in the proxy tar, 16 deterministic blank/ruled-paper/text-only negative control pages (`target = "" -> [BOS, EOS]`) are synthesized on disk so the `0.5%` Neg bucket is faithfully populated. If `rism_pseudo` rows are absent, excess weight spills cleanly into `synth_hw` + `real_hw_archival` (respecting the `<= 10×` cap in Arm 2) and is explicitly logged in telemetry.
- **OOD Eval Data:** `datasets/suite31_hw_eval.tar` — **all 45 pieces × 2 views (`clean` + `photo`) = 90 pages** via `suite31_hw_manifest.json` (`31` printed + `14` real historical handwritten).

## 4. Exact 3-Arm Controlled Protocol & Hyperparameters

Launched via `experiments/overnight_queue/run_queue_colab.py` or standalone:

```bash
python -u experiments/2026-11-27T08-19-i15-09-data-mix-ratio-and-rep-cap-sweep/run_colab.py
```

### Shared Controls (Identical Across All 3 Arms)
- **Warm-start weights:** Exact same frozen `UniOMRv2` EMA state dict (verified by a 2-page fail-fast printed sanity check before Arm 0).
- **Steps / Batch:** `800` optimizer steps/arm (`grad_accum=4` → `3,200` page draws/arm), `SEED=42`.
- **Optimizer & Schedule:** `AdamW(lr=5e-5, weight_decay=0.01, betas=(0.9, 0.95))`, `80`-step linear warmup, WSD cosine cooldown over final `25%` (`min_lr_ratio=0.10`), grad-norm clip `1.0`.
- **Precision & Input:** Native `BF16` (`torch.amp.autocast("cuda", dtype=torch.bfloat16)` on `cc >= 8`, FP16/FP32 fallback otherwise), canonical `[0, 1]` RGB normalization via `uni_omr._resize_fullpage_pil(..., 1280)` + `uni_omr._pil_to_chw_tensor`, `MAX_SEQ_LEN=1536`, `EVAL_MAX_LEN=3584` (`1408` eval max dim).
- **EMA:** `GPU_EMA_DECAY = 0.99` updated every step (`0.99^800 = 3.2e-4`, zero step-0 lag trap). **Both RAW and EMA** checkpoints are evaluated on all 90 pages of `suite31_hw`.

### Isolated Variable: Data-Mix Ratio & Repetition Cap

| Arm | Printed | Synth-HW | Real-HW (`real_hw_archival`) | RISM-Pseudo | Real-Stitched | Neg | Special Policy |
|---|---:|---:|---:|---:|---:|---:|---|
| `arm0_v3_50_50` | `50.0%` | `19.5%` | `12.0%` | `12.0%` | `6.0%` | `0.5%` | Baseline Iter 14 `v3` mix (no rep cap, uniform intra-bucket) |
| `arm1_hw_heavy_30_70` | `30.0%` | `25.0%` | `18.0%` | `20.0%` | `6.5%` | `0.5%` | HW-heavy `30/70` mix |
| `arm2_rep_cap_10x_hard_upsample` | `50.0%` | `19.5%` | `12.0%` *(capped)* | `12.0%` *(+spill)* | `6.0%` | `0.5%` | Per-unique-page cap `<= 10×` on `real_hw_archival` (excess spills cleanly to `rism_pseudo` + `synth_hw`) + `2×` upsampling of hard layouts (landscape 4-staff, lead-sheet/lyrics, dense 5+ staff) |

**Hard-layout detector (`arm2` `2×` intra-bucket weight):**
1. **Landscape 4-staff:** image aspect ratio `W / H >= 1.15` and `n_systems >= 4` (or `W > H` with `n_systems >= 3`).
2. **Lead-sheet / lyrics:** path/domain/metadata or tokens match lyrics/vocal/lead-sheet markers (`lyr`, `lyric`, `lieder`, `vocal`, `lead`, `choral`, `harm`).
3. **Dense 5+ staff:** `n_systems >= 5` (`len(bekerns) >= 5`).

## 5. 5-Minute Checkpointing & Resume Plan

- Atomic checkpointing (`.tmp` → `os.replace`) every `5.0` minutes (`SAVE_EVERY_MINUTES = 5.0`) to `<vol>/out/2026-11-27T08-19-i15-09-data-mix-ratio-and-rep-cap-sweep/<arm>/ckpt_latest.pt` (storing `step`, `model`, `ema`, `opt`, `draw_counts`, `bucket_draws`, `spilled_draws`, `loss_hist`).
- Completed arms write atomic `<arm>/arm_done.json` so re-launching `run_colab.py` after preemption skips finished arms and resumes mid-arm cleanly.
- Final `evals/<EXP_ID>.json` and `runs/<EXP_ID>.json` are uploaded to HF Hub (`gregoreesmaa/sheet-music-converter-omr`).

## 6. Evaluation Gate & Decision Tree

Evaluated on **all 90 pages** of `suite31_hw` (`printed_clean`, `printed_photo`, `printed_all`, `hw_clean`, `hw_photo`, `hw_all`, `all90`, plus `hw_landscape` / `hw_portrait`), checking **both RAW and EMA** checkpoints for every arm (using the best of RAW/EMA per arm, while flagging any `> 1.5 pp` RAW-vs-EMA divergence in scrutiny):

- **PROMOTE:**
  - Candidate arm (`arm1_hw_heavy_30_70` or `arm2_rep_cap_10x_hard_upsample`) achieves **`hw_all` Note F1 `>= +2.0 pp`** vs `arm0_v3_50_50` **AND** **`printed_all` Note F1 drop `<= 0.5 pp`** (`printed_delta_pp >= -0.5`).
  - Action: Promote the winning mix / repetition-cap / hard-layout upsampler into `tmp-train/mvd/train_uni_omr_v2.py` for Iter 15.
- **ITERATE:**
  - Either: (a) `arm1` gains `>= +2.0 pp` HW Note F1 but drops Printed `> 0.5 pp` (queue intermediate `40/60` mix + `10×` rep-cap follow-up), OR (b) `arm2` gains `+0.5` to `+2.0 pp` HW Note F1 with `<= 0.5 pp` Printed drop (queue combined `40/60` + `<= 8×` rep-cap + `2.5×` hard-layout upsample follow-up).
- **DISCARD:**
  - Both `arm1` and `arm2` gain `< +0.5 pp` HW Note F1 over `arm0_v3_50_50`, OR both regress `printed_all` by `> 1.0 pp` without a compensating `>= +2.0 pp` HW gain.
  - Action: Retain Iter 14 `v3` `50/50` mix as default.
