# Iter 13 — `UniOMR v2` (`small_45m`, Promoted Recipe + All Iter-13 Bug Fixes)

- **Experiment ID:** `2026-11-26T15-00-iter13-v2-promoted-recipe`
- **Status:** `COMPLETED` — **`PROMOTE`** (see [OUTCOME.md](./OUTCOME.md))
- **Baseline Commit SHA:** `36484e2` (Iter-13 bug fixes landed in `f089d62`)
- **Platform / GPU:** Local `RTX 4080 SUPER` (`16 GB` VRAM, BF16 native, `$0` cloud cost)
- **Quick Links:**
  - Plan & Decision Tree: [PLAN.md](./PLAN.md)
  - Verified 2-Stage + Eval Launcher: [launch_iter13.ps1](./launch_iter13.ps1)
  - Outcome *(written on completion)*: `OUTCOME.md`
  - HF Repo Paths (`gregoreesmaa/sheet-music-converter-omr`):
    - Ledger: `runs/2026-11-26T15-00-iter13-v2-promoted-recipe.json`
    - Checkpoints: `checkpoints/2026-11-26T15-00-iter13-v2-promoted-recipe/`
    - Eval JSONs: `evals/2026-11-26T15-00-iter13-v2-promoted-recipe_suite31_hw.json`
- **Local Output Dirs:**
  - Stage 1 (`4,000` steps strip warmup): `D:\tmp\mvd-runs\uni-omr-iter13-v2-stage1-strip4k`
  - Stage 2 (`24,000` steps pure fullpage + INT4 QAT): `D:\tmp\mvd-runs\uni-omr-iter13-v2-stage2-24k`
  - `suite31_hw` Eval Reports: `D:\tmp\uni-omr-suite31-ood\iter13_v2_stage2_24k_ema_*.json`

---

## Summary & Hypothesis

Iter 13 integrates **every promoted cloud ablation win** (`ov-14`, `ov-21`, `ov-22`, `ov-24`, `ov-34b`, `ov-35b`, `ov-39`, `i13-05`, `i13-10`, `i13-17`) and **every forensic P0/P1 bug fix** (`f089d62`) into a clean 2-stage `UniOMR v2 small_45m` (`~43M` params, `~28 MB` WebGPU INT4) training + evaluation run:

1. **Stage 1 (`4,000` steps strip warmup)** with decoupled `Muon` LR (`1.0e-3` / `3.0e-4`), explicit `--strip-grad-accum 4`, HW-safe `v1` KD (`--kd-teacher-ce-cap 1.5 --kd-hw-scale 0.0`), and `--mixstyle-prob 0.0`.
2. **Stage 2 (`24,000` steps pure fullpage, `--curriculum fullpage --fullpage-frac 1.00`)** at `1280px` multi-scale with cooled warm-start LRs (`--muon-lr 4.0e-4 --lr 2.5e-4`), WSD `0.20` cooldown, online unpaired RISM-5707 FDA + Color-AdaIN domain transfer (`i13-05`), stratified archival vs stitched real-HW sampling (`--split-archival-hw`) + dedicated real-HW val (`--val-real-hw-count 16`), and unrotated last-15% `INT4 QAT`.

**Hypothesis:** Eliminating `v1` teacher HW poisoning (`--kd-hw-scale 0.0 --kd-teacher-ce-cap 1.5`), disabling harmful `MixStyle2d` (`--mixstyle-prob 0.0`), removing Stage-2 strip dilution (`--fullpage-frac 1.00`), adding unpaired RISM FDA/AdaIN sim-to-real transfer, and training on top of the P0 router (`p10 + 0.15`, `luma_stop=0.35`, `vrfm_band_alpha=0.30`) and 4-loophole grammar fixes will push `UniOMR v2` past the `Iter 10.5` `v1` all-time `suite31_hw` record (**`all45_clean=0.6984`**, **`all45_photo=0.6832`**, **`hw_clean=0.2591`**, **`hw_photo=0.2174`**) and decisively beat `Iter 12` Stage 2 (`all45_clean=0.6555`, `all45_photo=0.6114` at step 8k EMA + P0 fixes) at `~28 MB` WebGPU INT4.

---

## Complete Provenance Table (Every Integrated Win & Bug Fix)

| Category | Change / Flag | Source Experiment / Commit | Status in Code / CLI | Measured Evidence / Rationale |
|---|---|---|---|---|
| **P0 Router & Paper-Gate** | `p10 + 0.15` adaptive saliency threshold (`InkTokenRouterV2`) | Forensic P0 (`f089d62`, `tmp-train/mvd/uni_omr_v2.py`) | Baked into `uni_omr_v2.py` | Prevents sparse/aged-parchment ink drop under global-mean shift |
| **P0 Router & Paper-Gate** | `luma_stop=0.35`, `±6px` border dilation (`1×13 + 13×1`) | `PROMOTE` [`ov-21`](../2026-11-26T01-38-ov-21-paper-gate-border-threshold/OUTCOME.md) + `f089d62` | Baked into `uni_omr_v2.py:864-881` | Worst-page RISM ink recall `+0.2609` (`0.6341 → 0.8950`), `0` collapsed pages, desk suppression `>= 0.96` |
| **P0 Router & Paper-Gate** | Horizontal staff-band saliency smoothing (`vrfm_band_alpha=0.30`) | `PROMOTE` [`i13-17`](../2026-11-26T07-40-i13-17-paddleocr-vrfm-band-router/OUTCOME.md) + `f089d62` | Baked into `uni_omr_v2.py:277,808,917` | `hw_clean +0.0142`, `hw_photo +0.0097` (`+0.0120` HW avg), **`+0.0000` printed regression**, `0 ms` latency |
| **P0 Grammar** | 4-loophole loop-breaker (`MAX_TOKENS_PER_BAR=288`, `RECENT_WINDOW_KEEP=576`, `bar_ngram_max=160`, empty trailing `<sys>` guard) | Forensic P0 (`f089d62`, `tmp-train/mvd/grammar.py`) | Baked into `grammar.py` (`test_grammar_loop_fixes.py`) | Closes long-bar repetition, window-eviction, cross-bar n-gram, and empty-`<sys>` decode loops |
| **P1 Training Fixes** | `--kd-teacher-ce-cap 1.5 --kd-hw-scale 0.0` | Forensic P1 (`f089d62`, `train_uni_v2.py`) | Explicit CLI in Stage 1 & Stage 2 | Stops `v1` teacher (weak on HW) from poisoning HW batches or high-CE tokens |
| **P1 Training Fixes** | `--strip-grad-accum 4` | Forensic P1 (`f089d62`, `train_uni_v2.py`) | Explicit CLI in Stage 1 | Fixes legacy auto-formula strip grad-accum mismatch; guarantees `eff_batch = 8 × 4 = 32` |
| **P1 Training Fixes** | `--split-archival-hw` + `--val-real-hw-count 16` | Forensic P1 (`f089d62`, `train_uni_v2.py`) | Explicit CLI in Stage 2 | Separates archival (`10%`) vs stitched (`6%`) real HW in sampler + adds dedicated `fullpage_hw_val` (`16` pages) |
| **P1 Training Fixes** | `--no-decay-1d --grad-clip 1.0` + dead aux-head skip | Forensic P1 (`f089d62`, `train_uni_v2.py`) | Explicit CLI + baked into `train_uni_v2.py` | Excludes 1D/0D norms/biases/embeddings from AdamW decay, stabilizes Muon+AdamW, skips zero-weight aux heads |
| **Promoted Domain Transfer** | `--rism-style-dir D:\tmp\unlabeled-10k-real\rism_hw --rism-fda-prob 0.25 --rism-adain-prob 0.20 --rism-fda-beta 0.02` | `PROMOTE` [`i13-05`](../2026-11-26T07-40-i13-05-fda-adain-rism-sim2real-transfer/OUTCOME.md) + `f089d62` | Explicit CLI in Stage 2 | Unpaired RISM-5707 transfer (`0` pseudo-label noise): FDA `β=0.02` `+0.0165` HW avg, Color-AdaIN `+0.0155` HW avg, `<3.5%` overhead |
| **Promoted Curriculum** | Stage 1 `4k` strip warmup → Stage 2 `--curriculum fullpage --fullpage-frac 1.00` (`24k` steps) | `PROMOTE` [`ov-22`](../2026-11-26T01-38-ov-22-fullpage-frac-sweep/OUTCOME.md) | Explicit CLI in Stage 2 | `fp_100` beats `0.70`/`0.85` monotonically on both domains (`+0.0389` Raw F1, `-0.0247` Val CE, highest `sys_count_exact`) |
| **Promoted Regularization** | `--mixstyle-prob 0.0` (OFF!) | `DISCARD` MixStyle [`ov-24`](../2026-11-26T01-38-ov-24-mixstyle-domain-invariance/OUTCOME.md) | Explicit CLI in Stage 1 & Stage 2 | `ms_00` beat `ms_20`/`ms_40` on both printed (`+0.0477` F1) and zero-shot HW (`+0.0343` to `+0.0471` F1) |
| **Promoted Schedule & Warm-Start** | `--lr-schedule wsd --wsd-decay-frac 0.20` + cooled Stage 2 (`--muon-lr 4.0e-4 --lr 2.5e-4 --cass-prob 0.15`) | [`ov-14`](../2026-11-26T01-38-ov-14-wsd-cooldown-shape/OUTCOME.md) + `ov-13` / Exp 4 | Explicit CLI in Stage 2 | `ov-14` confirmed cosine WSD `0.15–0.20` beats `0.30`; cooled Stage 2 LR prevents warm-start shock |
| **Fixed Online Augment** | `do_archival` / `do_1bit` geometry + dark-borders + lighting + curl/shadow/glare/thin-ink | Forensic P1 (`f089d62`, `tmp-train/mvd/uni_omr.py`) | Baked into `uni_omr.py` (`--augment --aug-prob 0.75`) | Fixes archival/1-bit augmentation bypasses so photo/degraded robustness covers all strata |
| **Promoted Inference (Fast)** | `--max-dim 1408 --max-len 3584 --use-grammar --use-mtp-spec` | `PROMOTE` [`ov-35b`](../2026-11-26T02-45-ov-35b-lower-res-tight-maxlen-sweep/OUTCOME.md) | Explicit CLI in Eval Pass 1 | Global 2D peak (`0.7016` clean / `0.6833` photo, `+0.0144` 90p mean over `1536/3840`) |
| **Promoted Inference (High-Acc)** | `+ --oemer-dewarp --measure-beam-size 2 --measure-beam-cap 4 --measure-beam-lp 2.0 --measure-beam-max-window 64` (plus opt-in fixed `--unitsize-norm`) | `PROMOTE` [`ov-34b`](../2026-11-26T05-35-ov-34b-budgeted-logprob-measure-beam/OUTCOME.md) + `PROMOTE` [`i13-10`](../2026-11-26T07-40-i13-10-oemer-staff-dewarp-unitsize/OUTCOME.md) | Explicit CLI in Eval Pass 2 (`eval_uni_suite31.py`) | `ov-34b`: `+0.0104` `pr_clean`, `+0.0081` `all45_clean` (`+30.2%` latency); `i13-10`: `+0.0335` `hw_photo`, `+0.0356` `all45_photo` (`78–128 ms/page` CPU) |
| **Promoted Data Pipeline Note** | RISM pseudo-label manifest builder (`mean_logprob >= -0.35`, `beat_valid_frac >= 0.70` / `0.85`) | `PROMOTE` [`ov-39`](../2026-11-26T01-38-ov-39-rism-pseudolabel-threshold-sweep/OUTCOME.md) | `tmp-train/mvd/build_rism_pseudolabel_manifest.py` | Ready for Iter-14 semi-supervised expansion (`~1,484` / `~1,027` high-yield RISM pages from 5,707 pool) once Iter-13 `v2` or `Iter 10.5` full pseudo-label pass lands |

---

## Baselines to Beat (`suite31_hw`, 45 Pieces = 90 Pages)

| Checkpoint | Eval Config | `printed_clean` | `printed_photo` | `hw_clean` | `hw_photo` | **`all45_clean`** | **`all45_photo`** |
|---|---|---:|---:|---:|---:|---:|---:|
| **Iter 10.5 `v1` EMA** *(all-time record & KD teacher)* | `1536px / 4096`, `FSA+MTP` | `0.8968` | `0.8935` | `0.2591` | `0.2174` | **`0.6984`** | **`0.6832`** |
| **Iter 12 `v2` Stage 2 step 8k EMA** *(+ P0 fixes)* | `1408px / 3584`, `FSA+MTP` | `0.8728` | `0.8369` | `0.1744` | `0.1121` | `0.6555` | `0.6114` |
| **Iter 13 `v2` Target (`PROMOTE` gate)** | `1408px / 3584`, `FSA+MTP` | `>= 0.900` | `>= 0.895` | `>= 0.275` | `>= 0.230` | **`>= 0.705`** | **`>= 0.690`** |
