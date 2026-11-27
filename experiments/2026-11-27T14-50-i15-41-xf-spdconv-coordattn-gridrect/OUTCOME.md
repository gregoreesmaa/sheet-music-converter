# OUTCOME — `2026-11-27T14-50-i15-41-xf-spdconv-coordattn-gridrect` (`i15-41`)

- **Status:** `COMPLETED` (`rc=0`, `2/2` arms completed, zero NaNs, `error: null`)
- **Verdict:** **`DISCARD`**
- **Platform / GPU:** Local Windows 11 — `NVIDIA GeForce RTX 4080 SUPER` (16 GB VRAM, native BF16, `--no-cudnn` + CUTLASS `EFFICIENT_ATTENTION`)
- **Started / Finished (UTC):** `2026-11-27T14:24:23Z` → `2026-11-27T15:12:51Z` (`2908.34 s` / `48.5 min` total)
- **Baseline Commit SHA:** `6055c49`
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-27T14-50-i15-41-xf-spdconv-coordattn-gridrect.json` (pulled to [`eval_hf.json`](eval_hf.json))
  - Run Ledger: `runs/2026-11-27T14-50-i15-41-xf-spdconv-coordattn-gridrect.json` (pulled to [`run_hf.json`](run_hf.json))

---

## Measured Results (`ctrl` vs `spd_coord_rect`, 600 steps/arm, `muon+adamw`, `seed=42`)

| Metric | `ctrl` (stock `AnisotropicMusicBackbone`) | `spd_coord_rect` (`SPD-Conv` + `CoordAttn2d` + `GridRect`) | Delta (`spd_coord_rect - ctrl`) | Gate (`PLAN.md`) | Pass? |
|---|---:|---:|---:|---|---|
| **`ema_val_ce`** *(primary)* | **1.12726** | 1.12895 | **+0.00169** (worse) | `<= -0.015` PROMOTE / `<= -0.005` ITERATE | **FAIL** |
| **`int4_dequant_val_ce`** *(primary)* | **1.12994** | 1.13142 | **+0.00148** (worse) | `<= -0.015` PROMOTE | **FAIL** |
| **`raw_val_ce`** | **1.10474** | 1.10623 | **+0.00149** (worse) | lower is better | **FAIL** |
| `train_loss_tail50` | **1.53757** | 1.54052 | +0.00295 (worse) | lower is better | **FAIL** |
| `int4_dequant - ema` gap | +0.00268 | +0.00247 | -0.00021 | `<= +0.025` | PASS |
| **`step_ms`** *(steady mean, steps 11–600)* | **1641.99 ms** | 2798.95 ms | **+1156.96 ms (+70.5%)** | `<= +12%` | **FAIL** |
| `step_ms` *(pre-QAT, steps 10–300)* | 1646.9 ms | 2033.1 ms | +386.2 ms (**+23.5%**) | `<= +12%` | **FAIL** |
| `step_ms` *(post-QAT, steps 420–568)* | 1638.7 ms | 4119.1 ms | +2480.4 ms (**+151.4%**) | `<= +12%` | **FAIL** |
| `train_elapsed_sec` | 987.43 s | 1672.15 s | +684.72 s (+69.3%) | — | — |
| **`peak_vram_mb`** | **11,046.1 MB** | 12,079.5 MB | **+1,033.4 MB (+9.4%)** | `< 14,000 MB` WDDM safe | WARN (paging) |
| **`packed_int4_mb`** *(exact disk bytes)* | **53.039 MB** | 53.915 MB | +0.876 MB (`+438,201` FP16 params) | `< 55.0 MB` hard cap | PASS |
| `grid_rect_mean_abs_dy` | N/A | **0.000174** | ~0.11 px on 1280 px page (dead) | active non-trivial warp | **DEAD** |
| **Overall Gate (`gate_pass`)** | baseline | — | — | all primary gates | **FAIL → `DISCARD`** |

---

## Scrutiny & Anomaly Check

1. **Silent failures / partial coverage:** None. Both arms completed all `600/600` steps with verified `optimizer_mode == "muon+adamw"`, `bf16 == true`, `seed == 42`, zero NaNs, and exact on-disk Hybrid INT4 binary packing (`53,039,072 B` vs `53,915,474 B`).
2. **Apples-to-apples sanity:** Identical `iter10_5_proxy_2k.tar` dataset (`64` val pages, `FULLPAGE_MAX_DIM=1280`, `MAX_SEQ_LEN=1536`), identical `edge85m + ax0_gl6` backbone/encoder/decoder (`71,438,848` INT4 weights identical across both arms), identical WSD schedule and `QATInt4ControllerV2` enablement at step `420`. Note that `raw_val_ce` (`1.10474`) is ~0.0225 lower than `ema_val_ce` (`1.12726`) in both arms because `EMA_DECAY=0.99` lags slightly at 600 steps; both raw and EMA agree on the exact same `~+0.0015` to `+0.0017` regression for `spd_coord_rect`.
3. **Root Cause 1 — Zero accuracy lift across all metrics (`+0.00169` EMA CE, `+0.00148` INT4 CE):**
   - Stock `AnisotropicMusicBackbone` already preserves high-resolution `/4` features via `p2_unshuffle` (`PixelUnshuffle(2)` skip projections into `stage3` and `stage4`). Replacing `stage1.down` (`/1→/2`) and `stage2.down` (`/2→/4`) with `SPDConvDown` (`4×` channel expansion + non-strided `3×3` conv) + `CoordAttn2d` added `+438,201` FP16-kept parameters (`n_fp16_tensors: 216 → 245`, `+0.876 MB` packed) without improving token cross-entropy at any point in the training curve (losses tracked `ctrl` within noise from step 10 through step 600: `3.66 vs 3.67` → `2.31 vs 2.31` → `2.00 vs 1.99` → `1.78 vs 1.78` → `1.64 vs 1.64` → `1.54 vs 1.54`).
   - **`VerticalGridRectifier` remained dead at identity (`grid_rect_mean_abs_dy = 0.000174`):** On `[-1, 1]` normalized coordinates (`max_shift = 0.12`), `0.000174` corresponds to `0.0087%` of page height (`~0.11 px` at 1280 px). Zero-initialized `delta_head` combined with `RECT_SMOOTH_WEIGHT = 0.05` and weak indirect CE gradients through `F.grid_sample` never moved the displacement field off zero.
4. **Root Cause 2 — Hardware/WDDM anomaly (`step_ms` jumped `+70.5%` overall, `+151.4%` post-QAT):**
   - Inspecting `step_ms_hist` from `D:\tmp\local-p0\i15-41\ckpts/{ctrl,spd_coord_rect}/latest.pt`:
     - **`ctrl`:** rock-steady across all 600 steps — `1644 ms` (10–100), `1648 ms` (100–200), `1649 ms` (200–300), `1639 ms` (300–420), `1647 ms` (420–500 post-QAT), `1624 ms` (500–548).
     - **`spd_coord_rect`:** already **+23.5% slower pre-QAT** (`2003 ms` steps 10–100, `2028 ms` steps 100–200, `2068 ms` steps 200–300) due to non-strided `3×3` convs over `4×`-channel unshuffled high-res `/2` and `/4` maps under `--no-cudnn`, then climbed to `2429 ms` (steps 300–420) and **exploded to `4015.5 ms` (steps 420–500) and `4241.1 ms` (steps 500–568, max `7681.8 ms`)** the moment `QATInt4ControllerV2` turned on at step `420`.
   - **Mechanism:** `spd_coord_rect` raised peak allocated VRAM by `+1,033.4 MB` (`11,046.1 MB → 12,079.5 MB`). With PyTorch's caching allocator reserve + QAT fake-quantization temporaries + Windows 11 desktop WDDM overhead (~1.5–2 GB), total GPU reservation crossed the 16 GB RTX 4080 SUPER WDDM cliff and silently spilled to shared system RAM over PCIe, causing a ~2× step-time stall during the final 30% QAT phase.
   - **Important:** Even excluding the WDDM paging cliff, `spd_coord_rect` already failed the `<= +12%` latency gate at `+23.5%` pre-QAT and failed the accuracy gate on every metric.

---

## Verdict & Follow-Up Action

- **Verdict:** **`DISCARD`** — falsified hypothesis. Do **not** port `SPDConvDown`, `CoordAttn2d`, or unsupervised `VerticalGridRectifier` into `AnisotropicMusicBackbone` in `tmp-train/mvd/uni_omr_v2.py`.
- **Keep:** Stock `AnisotropicMusicBackbone` (`53.039 MB` packed INT4, `1642 ms/step`, `11.05 GB` peak VRAM, `1.12726` EMA CE / `1.12994` INT4 dequant CE).
- **Operational Takeaway for Local 4080 SUPER Proxies:** When `peak_vram_mb` (allocated) approaches `~12.0 GB` on Windows 11 16 GB WDDM, enabling `QATInt4ControllerV2` at step 420 can push reserved VRAM past the WDDM paging cliff (2× slowdown). Keep local 4080 SUPER proxy arms `<= 11.2 GB` peak allocated VRAM or run heavier backbone probes on Colab A100/L4.
- **Follow-Up / Successor:** None required for this branch (`DISCARD` terminal). Proceed with the remaining independent Wave-0 cross-field proxies in the queue.
