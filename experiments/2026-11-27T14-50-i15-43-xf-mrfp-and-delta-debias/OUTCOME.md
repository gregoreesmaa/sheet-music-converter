# OUTCOME — `2026-11-27T14-50-i15-43-xf-mrfp-and-delta-debias` (`i15-43`)

- **Status:** `COMPLETED` (`rc=0`, `2/2` arms completed, zero NaNs, `error: null`)
- **Verdict:** **`DISCARD`**
- **Platform / GPU:** Local Windows 11 — `NVIDIA GeForce RTX 4080 SUPER` (16 GB VRAM, native BF16, `--no-cudnn` + CUTLASS `EFFICIENT_ATTENTION`)
- **Started / Finished (UTC):** `2026-11-27T17:11:28Z` → `2026-11-27T18:15:56Z` (`3867.84 s` / `64.5 min` total)
- **Baseline Commit SHA:** `6055c49`
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-27T14-50-i15-43-xf-mrfp-and-delta-debias.json` (pulled to [`eval_hf.json`](eval_hf.json))
  - Run Ledger: `runs/2026-11-27T14-50-i15-43-xf-mrfp-and-delta-debias.json` (pulled to [`run_hf.json`](run_hf.json))

---

## Measured Results (`ctrl` vs `mrfp_delta`, 600 steps/arm, `muon+adamw`, `seed=42`)

| Metric | `ctrl` (stock backbone) | `mrfp_delta` (`MRFP` train + `DELTA` SVD fold) | Delta (`mrfp_delta - ctrl`) | Gate (`PLAN.md`) | Pass? |
|---|---:|---:|---:|---|---|
| **`ema_val_ce`** *(clean, pre-DELTA)* | **1.12715** | 1.12871 | **+0.00156** (worse) | `<= +0.005` (with degraded gain) | **FAIL** |
| **`ema_val_ce_degraded`** *(pre-DELTA)* | **1.12888** | 1.13060 | **+0.00172** (worse) | `<= -0.020` PROMOTE / `<= -0.005` | **FAIL** |
| **`delta_val_ce_clean`** *(post-DELTA, `k=4`)* | 1.12814 *(+0.00099 vs pre)* | 1.12856 *(-0.00015 vs pre)* | **+0.00141** *(vs ctrl pre-DELTA)* | `<= +0.005` | **FAIL** |
| **`delta_val_ce_degraded`** *(post-DELTA, `k=4`)* | **1.12857** *(-0.00031 vs pre)* | 1.12955 *(-0.00105 vs pre)* | **+0.00067** *(vs ctrl pre-DELTA)* | `<= -0.020` PROMOTE / `<= -0.005` | **FAIL** |
| **`int4_dequant_val_ce`** *(clean)* | **1.12921** | 1.13052 | **+0.00131** (worse) | lower is better | **FAIL** |
| **`int4_dequant_val_ce_degraded`** | **1.13042** | 1.13154 | **+0.00112** (worse) | `<= -0.020` PROMOTE | **FAIL** |
| **`raw_val_ce`** *(clean)* | **1.10426** | 1.10645 | +0.00219 (worse) | lower is better | **FAIL** |
| `train_loss_tail50` | **1.53752** | 1.54007 | +0.00255 (worse) | lower is better | **FAIL** |
| `int4_dequant - ema` gap *(clean)* | +0.00206 | +0.00181 | -0.00025 | `<= +0.025` | PASS |
| **`step_ms`** | **1939.76 ms** | 4089.35 ms | **+2149.59 ms (+110.8%)** | `<= +8%` | **FAIL** |
| `train_elapsed_sec` | 1164.14 s | 2453.52 s | +1289.38 s (+110.8%) | — | — |
| **`peak_vram_mb`** | **11,124.5 MB** | 12,862.2 MB | **+1,737.7 MB (+15.6%)** | `< 14,000 MB` WDDM safe | **WARN (paging)** |
| **`packed_int4_mb`** *(exact disk bytes)* | **53.039 MB** | **53.039 MB** | **0.000 MB** (`53,039,072 B`) | `< 55.0 MB` & `+0.000 MB` | PASS |
| **Overall Verdict** | baseline | — | — | degraded `<= -0.005` & `step_ms <= +8%` | **FAIL → `DISCARD`** |

### `DELTA` SVD Subspace Sweep (Holdout 32 Val Pages, `k ∈ {2, 4, 8, 16}`)

| `k` | `ctrl` Holdout Clean CE | `ctrl` Holdout Degraded CE | `mrfp_delta` Holdout Clean CE | `mrfp_delta` Holdout Degraded CE |
|---:|---:|---:|---:|---:|
| `2` | 1.12122 | 1.12164 | 1.12309 | 1.12416 |
| **`4`** *(best)* | **1.12092** | **1.12124** | **1.12254** | **1.12298** |
| `8` | 1.12664 | 1.12642 | 1.12883 | 1.12834 |
| `16` | 1.13346 | 1.13294 | 1.13906 | 1.13907 |

---

## Scrutiny & Anomaly Check

1. **Silent failures / partial coverage:** None. Both arms completed all `600/600` steps with verified `optimizer_mode == "muon+adamw"`, `bf16 == true`, `seed == 42`, zero NaNs, and exact `0.000 MB` packed Hybrid INT4 delta (`53,039,072 B`).
2. **Apples-to-apples sanity:** Identical `iter10_5_proxy_2k.tar` dataset (`64` val pages, paired clean + degraded), identical `edge85m + ax0_gl6` architecture, identical WSD + QAT at step `420`. `ctrl` clean EMA CE (`1.12715`) matches sibling Wave-0 baselines (`i15-41`: `1.12726`, `i15-42`: `1.12690`).
3. **Root Cause 1 — Neither `MRFP` nor `DELTA` yields a meaningful robustness gain:**
   - **`MRFP` (pre-DELTA)** regressed **both** clean (`+0.00156`) and degraded (`+0.00172`) validation CE as well as INT4 dequantized CE (`+0.00131` clean, `+0.00112` degraded). Because `iter10_5_proxy_2k` already includes heavy `photo-degrade` realism during training, the stock backbone is already largely invariant to photometric/illumination jitter (the paired degraded val split only shifted `ctrl` CE by `+0.00173`, `1.12715 → 1.12888`). Perturbing intermediate feature statistics with `MRFP` merely added optimization noise.
   - **`DELTA` SVD projection** (`k=4` best, top-4 singular vectors explaining `63.5%` of paired difference variance on `ctrl` and `64.3%` on `mrfp_delta`) produced only a noise-level `-0.00031` shift on `ctrl` degraded CE while hurting `ctrl` clean CE by `+0.00099`. At `k >= 8`, projecting out subspaces actively destroyed notation features (`+0.006` to `+0.016` worse CE on holdout).
4. **Root Cause 2 — Hardware/WDDM anomaly (`step_ms` jumped `+110.8%`, `1939.76 → 4089.35 ms`):**
   - `MRFP`'s multi-stage feature-stat randomization and spatial envelope tensors increased peak allocated VRAM by **`+1,737.7 MB`** (`11,124.5 → 12,862.2 MB`). Combined with PyTorch caching allocator headroom + QAT temporaries + Windows 11 desktop WDDM overhead, this pushed total GPU reservation past the 16 GB RTX 4080 SUPER WDDM paging cliff, causing a `2.11×` slowdown (same WDDM cliff observed in `i15-41`).

---

## Follow-Up Action / Successor Experiment

- **Verdict:** **`DISCARD`** — falsified hypothesis. Do **not** integrate `MRFP` feature perturbation into `AnisotropicMusicBackbone` or post-training `DELTA` subspace projection into Iter 15.
- **Keep:** Stock `AnisotropicMusicBackbone` trained on realistic `photo-degrade` + real HW pages (`53.039 MB` packed INT4, `1.12715` clean / `1.12888` degraded EMA CE).
- **Successor:** None required (`DISCARD` terminal).
