# OUTCOME — `2026-11-27T14-50-i15-42-xf-aux-cbdice-unmix-gdcm` (`i15-42`)

- **Status:** `COMPLETED` (`rc=0`, `2/2` arms completed, zero NaNs, `error: null`)
- **Verdict:** **`DISCARD`**
- **Platform / GPU:** Local Windows 11 — `NVIDIA GeForce RTX 4080 SUPER` (16 GB VRAM, native BF16, `--no-cudnn` + CUTLASS `EFFICIENT_ATTENTION`)
- **Started / Finished (UTC):** `2026-11-27T16:32:39Z` → `2026-11-27T17:11:22Z` (`2323.20 s` / `38.7 min` total)
- **Baseline Commit SHA:** `6055c49`
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-27T14-50-i15-42-xf-aux-cbdice-unmix-gdcm.json` (pulled to [`eval_hf.json`](eval_hf.json))
  - Run Ledger: `runs/2026-11-27T14-50-i15-42-xf-aux-cbdice-unmix-gdcm.json` (pulled to [`run_hf.json`](run_hf.json))

---

## Measured Results (`ctrl` vs `aux_topo_unmix_gdcm`, 600 steps/arm, `muon+adamw`, `seed=42`)

| Metric | `ctrl` (stock CE + MTP-4 + router) | `aux_topo_unmix_gdcm` (+ `InkUnmix` + `cbDice` + `GDCM`) | Delta (`aux - ctrl`) | Gate (`PLAN.md`) | Pass? |
|---|---:|---:|---:|---|---|
| **`ema_val_ce`** *(primary)* | **1.12690** | 1.13125 | **+0.00435** (worse) | `<= -0.010` PROMOTE / `<= -0.003` ITERATE | **FAIL** |
| **`int4_dequant_val_ce`** *(primary)* | **1.12840** | 1.13328 | **+0.00488** (worse) | `<= -0.010` PROMOTE | **FAIL** |
| **`raw_val_ce`** | **1.10404** | 1.10890 | **+0.00486** (worse) | lower is better | **FAIL** |
| `train_loss_tail50` *(total incl. aux)* | **1.53806** | 1.62971 | +0.09165 *(base CE+MTP+router ≈ 1.54063, +0.00257)* | lower is better | **FAIL** |
| `aux_losses_tail50` | N/A | `unmix=0.11827`, `cbdice=0.25924`, `gdcm=0.51329` | weighted sum = `+0.08908` | diagnostic | INFO |
| `int4_dequant - ema` gap | +0.00150 | +0.00203 | +0.00053 | `<= +0.025` | PASS |
| **`step_ms`** | **1644.45 ms** | 1909.36 ms | **+264.91 ms (+16.1%)** | `<= +20%` (with CE gain) | WARN |
| `train_elapsed_sec` | 990.26 s | 1146.22 s | +155.96 s (+15.7%) | — | — |
| `peak_vram_mb` | **10,859.0 MB** | 12,045.3 MB | **+1,186.3 MB (+10.9%)** | `< 14,000 MB` | WARN |
| **`packed_int4_mb`** *(exact disk bytes, stripped)* | **53.039 MB** | **53.039 MB** | **0.000 MB** (`53,039,072 B`) | exact `0.000 MB` delta & `< 55.0 MB` | PASS |
| **Overall Verdict** | baseline | — | — | `ema_val_ce <= -0.003` | **FAIL → `DISCARD`** |

---

## Scrutiny & Anomaly Check

1. **Silent failures / partial coverage:** None. Both arms completed all `600/600` steps with verified `optimizer_mode == "muon+adamw"`, `bf16 == true`, `seed == 42`, zero NaNs, and exact `0.000 MB` stripped Hybrid INT4 delta (`53,039,072 B`, `131` INT4 + `216` FP16 tensors). `strip_aux_heads()` cleanly removed all `+864,259` training-only aux parameters (`total_params: 80,088,974 → 77,866,208` infer params).
2. **Apples-to-apples sanity:** Identical `iter10_5_proxy_2k.tar` dataset (`64` val pages, `FULLPAGE_MAX_DIM=1280`, `MAX_SEQ_LEN=1536`), identical `edge85m + ax0_gl6` architecture, identical WSD schedule and QAT at step `420`. `ctrl` (`1.12690` EMA CE, `1644.45 ms/step`) matches sibling Wave-0 baselines (`i15-41`: `1.12726` / `1641.99 ms`).
3. **Root Cause of `+0.00435` EMA CE / `+0.00488` INT4 CE Regression:**
   - Subtracting the weighted tail-50 auxiliary loss sum (`0.10 * (0.11827 + 0.25924 + 0.51329) = 0.08908`) from `train_loss_tail50` (`1.62971`) yields an underlying base train loss of `~1.54063` vs `1.53806` (`+0.00257` worse), matching the validation regression (`+0.00435` EMA CE, `+0.00486` raw CE, `+0.00488` INT4 CE).
   - **Negative multi-task gradient transfer:** `GDCM` global page-level counting (`0.51329` tail loss — by far the largest of the three) pulls pooled `mem_final` toward bag-of-symbols cardinality rather than fine-grained spatial alignment, while self-derived high-pass `InkUnmix` (`0.11827`) and 3-iter soft-skeleton `cbDice` (`0.25924`) compete with autoregressive notation CE gradients on `P2/P3` features.
4. **Hardware / resource impact:** Training overhead was `+16.1%` step time (`+264.91 ms/step`) and `+1,186.3 MB` peak VRAM (`10,859.0 → 12,045.3 MB`, right at the Windows 11 16 GB WDDM cliff) for a net accuracy loss.

---

## Follow-Up Action / Successor Experiment

- **Verdict:** **`DISCARD`** — falsified hypothesis. Do **not** add `InkUnmixHead`, `cbDice` soft-skeleton loss, or `GDCMCountHead` to `uni_omr_v2.py` / `train_uni_v2.py`.
- **Keep:** Stock training objectives (`CE + MTP-4 + router saliency`, `53.039 MB` packed INT4, `1644 ms/step`, `1.12690` EMA CE).
- **Successor:** None required (`DISCARD` terminal).
