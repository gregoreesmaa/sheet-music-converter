# OUTCOME — `2026-11-27T14-15-i15-14-3axis-structural-dec-rope` (`i15-14`)

- **Status:** `COMPLETED` (`rc=0`, `2/2` arms completed, zero NaNs, `error: null`)
- **Verdict:** **`DISCARD`**
- **Platform / GPU:** Google Colab — `NVIDIA A100-SXM4-40GB` (native BF16)
- **Started / Finished (UTC):** `2026-11-27T14:31:20Z` → `2026-11-27T15:23:12Z` (`3112.52 s` / `51.9 min` total)
- **Baseline Commit SHA:** `a32c810`
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-27T14-15-i15-14-3axis-structural-dec-rope.json` (pulled to [`eval_hf.json`](eval_hf.json))
  - Run Ledger: `runs/2026-11-27T14-15-i15-14-3axis-structural-dec-rope.json` (pulled to [`run_hf.json`](run_hf.json))

---

## Measured Results (`ctrl` vs `rope3d`, 600 steps/arm, `muon+adamw`, `seed=42`)

| Metric | `ctrl` (stock 1D Decoder RoPE) | `rope3d` (`StructuralRoPE3D` 16/16/32) | Delta (`rope3d - ctrl`) | Gate (`PLAN.md`) | Pass? |
|---|---:|---:|---:|---|---|
| **`ema_val_ce`** *(primary)* | **1.12761** | 1.16534 | **+0.03773** (worse) | `<= -0.015` PROMOTE / `<= -0.005` ITERATE | **FAIL** |
| **`int4_dequant_val_ce`** *(primary)* | **1.12885** | 1.16465 | **+0.03580** (worse) | `<= -0.015` PROMOTE | **FAIL** |
| **`raw_val_ce`** | **1.10556** | 1.13865 | **+0.03309** (worse) | lower is better | **FAIL** |
| `train_loss_tail50` | **1.53944** | 1.60368 | **+0.06424** (worse) | lower is better | **FAIL** |
| `int4_dequant - ema` gap | +0.00124 | -0.00069 | -0.00193 | `<= +0.025` | PASS |
| **`step_ms`** | 2375.81 ms | **2337.91 ms** | **-37.90 ms (-1.6%)** | `<= +5%` | PASS |
| `train_elapsed_sec` | 1437.42 s | 1409.71 s | -27.71 s (-1.9%) | — | — |
| `peak_vram_mb` | **10,625.4 MB** | 11,426.2 MB | +800.8 MB (+7.5%) | — | — |
| **`packed_int4_mb`** *(exact disk bytes)* | **53.039 MB** | **53.039 MB** | **0.000 MB** (`53,039,072 B`) | `< 55.0 MB` hard cap | PASS |
| **Overall Verdict** | baseline | — | — | `ema_val_ce <= -0.005` | **FAIL → `DISCARD`** |

---

## Scrutiny & Anomaly Check

1. **Silent failures / partial coverage:** None. Both arms completed all `600/600` steps on Colab `A100-SXM4-40GB` with verified `optimizer_mode == "muon+adamw"`, `bf16 == true`, `seed == 42`, zero NaNs, and bit-identical packed Hybrid INT4 artifact size (`53,039,072 B`, `131` INT4 tensors + `216` FP16 tensors).
2. **Apples-to-apples sanity:** Identical `iter10_5_proxy_2k.tar` dataset (`64` val pages, `FULLPAGE_MAX_DIM=1280`, `MAX_SEQ_LEN=1536`), identical `edge85m + ax0_gl6` parameter count (`77,866,208` infer params), identical `QATInt4ControllerV2` enablement at step `420`, and matching `ctrl` baseline (`1.12761` EMA CE vs `1.12690–1.12726` across sibling Wave-0 proxies `i15-41/42/43`).
3. **Root Cause of `+0.03773` EMA CE / `+0.06424` Train Loss Regression:**
   - Splitting `head_dim=64` into `(16, 16, 32)` subspaces for `(system_idx, measure_idx, intra_measure_pos)` **eliminates monotonic global 1D sequential distance** from decoder self-attention.
   - Because `intra_measure_pos` resets to `0` at every barline and `measure_idx` resets at every `<sys>`, local relative-position continuity across barlines and system breaks is severed. Autoregressive BeKern decoding relies heavily on smooth cross-barline continuity (ties, slurs, voice/spine alignment, clef/key/meter persistence).
   - Half of the rotary dimensions (`32/64`) are allocated to low-cardinality integer counters (`system_idx` ~`0..12`, `measure_idx` ~`0..8`), starving fine-grained token-to-token relative positioning.
4. **Hardware / platform quirks:** None on Colab A100-40GB (`step_ms` was steady and slightly faster on `rope3d`, `-1.6%`; VRAM `+800.8 MB` from 3-axis sin/cos index tensors remained well within 40 GB).

---

## Follow-Up Action / Successor Experiment

- **Verdict:** **`DISCARD`** — falsified hypothesis. Do **not** replace stock 1D decoder RoPE (`_CachedRoPE1D`) with `StructuralRoPE3D` in `tmp-train/mvd/uni_omr_v2.py`.
- **Keep:** Stock 1D sequential decoder RoPE (`53.039 MB` packed INT4, `1.12761` EMA CE / `1.12885` INT4 dequant CE).
- **Successor:** None required (`DISCARD` terminal).
