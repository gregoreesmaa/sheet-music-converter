# OUTCOME — `2026-11-27T10-30-i15-11-sheared-strip-conv-and-aniso-pitch-rope2d`

- **Status:** `COMPLETED`
- **Verdict:** **`DISCARD`**
- **Short ID / Wave:** `i15-11` (Iter 15 — Weight-Incompatible Architecture Proxy)
- **Started / Finished (UTC):** `2026-11-27T12:23:21Z` → `2026-11-27T13:07:46Z` (`2665.52 s` / `44.4 min` total wall-clock)
- **Platform / GPU:** Local `NVIDIA GeForce RTX 4080 SUPER` (`16 GB` VRAM, native BF16; executed via `D:\tmp\local-p0\i15-11`, `opt=muon+adamw` on both arms, same `seed=42` / `600` steps/arm protocol as `PLAN.md`)
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-27T10-30-i15-11-sheared-strip-conv-and-aniso-pitch-rope2d.json` (mirrored locally at [eval_hf.json](./eval_hf.json))
  - Run Ledger: `runs/2026-11-27T10-30-i15-11-sheared-strip-conv-and-aniso-pitch-rope2d.json` (mirrored locally at [run_hf.json](./run_hf.json))

---

## Measured Results (`ctrl` vs `sheared_aniso_rope`, 600 steps/arm, `iter10_5_proxy_2k`)

| Metric | `ctrl` (`edge85m + ax0_gl6`) | `sheared_aniso_rope` (`ShearedStripConv2d + PitchBiasedAnisoRoPE2D`) | Delta (`sheared - ctrl`) | `README.md` Gate | Status |
|---|---:|---:|---:|---|---|
| **`ema_val_ce`** (primary upright) | **`1.29295`** | `1.29325` | `+0.00030` (`+0.02%`) | `<= ctrl + 0.010` (ideally `< ctrl`) | PASS (flat / noise) |
| **`ema_val_tilted_ce`** (`±6°` tilt) | `1.29357` | **`1.29322`** | `-0.00035` (`-0.03%`) | `<= ctrl - 0.030` | **FAIL** (`85×` short) |
| **`tilt_degradation_ce`** (EMA `tilted - upright`) | `+0.00062` | `-0.00002` | `-0.00064` | — | Noise floor |
| **`val_ce`** (raw upright) | `1.08383` | **`1.07811`** | `-0.00572` (`-0.53%`) | — | Marginal (`< 0.01`) |
| **`val_tilted_ce`** (raw `±6°` tilt) | `1.08456` | **`1.07847`** | `-0.00609` (`-0.56%`) | — | Marginal (`< 0.01`) |
| **`raw tilt_degradation_ce`** (`tilted - upright`) | `+0.00073` | `+0.00036` | `-0.00037` | — | Noise floor |
| **`final_25step_loss`** | `1.7044` | **`1.6894`** | `-0.0150` | — | Marginal |
| **`int4_dequant_val_ce`** | *N/A* (not in runner) | *N/A* (not in runner) | — | — | Not measured |
| **`step_ms`** (train avg) | **`2041.87 ms`** | `2114.26 ms` | `+72.39 ms` (`+3.55%`) | `<= +8%` | PASS |
| **`elapsed_sec`** (train) | `1225.12 s` | `1268.55 s` | `+43.43 s` (`+3.55%`) | — | OK |
| **`peak_vram_mb`** | `11,081.9 MB` | `11,504.2 MB` | `+422.3 MB` (`+3.81%`) | `< 14,000 MB` | PASS |
| **`packed_int4_mb`** (analytic `param_summary`) | `53.039 MB` | `53.194 MB` | `+0.155 MB` (`+0.29%`) | `<= 55.0 MB` | PASS (`1.806 MB` headroom) |
| **`infer_params`** | `77,866,208` | `77,943,644` | `+77,436` (`+0.10%`) | — | OK |
| **`backbone_params`** | `5,178,140` | `5,255,576` | `+77,436` (`+1.50%`) | — | OK |
| **`val_tokens`** | `53,110` | `53,110` | `0` | Exact match | PASS |
| **`gate_pass`** | — | **FALSE** | — | All `README.md` gates | **FAIL** |

---

## Scrutiny & Anomaly Check

1. **Silent failures / coverage:** Both arms (`ctrl`, `sheared_aniso_rope`) completed all `600/600` steps with `rc=0`, `error: null`, zero NaNs, `opt=muon+adamw` on both arms, and identical `val_tokens = 53,110` across all `64` val pages (`32` batches). Downloaded HF `eval_hf.json` (`1,944 B`) and `run_hf.json` (`455 B`) match local `D:\tmp\local-p0\i15-11\out\2026-11-27T10-30-i15-11-sheared-strip-conv-and-aniso-pitch-rope2d\{eval,run}.json` byte-for-byte.
2. **Apples-to-apples sanity:** Both arms ran sequentially on the same local `RTX 4080 SUPER` (BF16) with identical `seed=42`, identical `iter10_5_proxy_2k` manifest, identical sampling RNG stream (`random.Random(42)` reset per arm), identical Muon+AdamW (`lr=3e-4`, `wd=0.01`, WSD schedule), and identical `±6°` deterministic alternating tilt stress eval (`_tilt_batch_images`).
3. **Anomaly #1 — Severe EMA lag (`ema_val_ce` `+0.21` nats worse than raw `val_ce`):** `run_colab.py` used constant `GPU_EMA_DECAY = 0.997` from step 1 with **no EMA warmup**. Over only `600` steps, `0.997^600 = 0.165` — **16.5% of the step-0 random initialization** remained blended into `ema.shadow` at evaluation time (`ema_val_ce ≈ 1.293` vs raw `val_ce ≈ 1.084`). This heavily damped any difference in EMA metrics (`+0.00030` upright / `-0.00035` tilted). **Crucially, even on un-lagged raw weights, the gain is still too small to justify the complexity and overhead:** `val_ce` improved by only `-0.00572` nats (`-0.53%`) and `val_tilted_ce` by `-0.00609` nats (`-0.56%`), less than one-quarter of genuine Iter-15 architectural wins on the same 600-step proxy (e.g., `i15-38` `coordkey_ca`: `-0.02635` raw / `-0.02014` EMA while `-1.42%` faster).
4. **Anomaly #2 — Near-zero `±6°` tilt degradation (`+0.00073` nats raw on `ctrl`):** Rotating full-page val images by `±6°` degraded `ctrl` CE by only `+0.00073` nats (`1.08383 → 1.08456`), making the `README.md` gate (`>= 0.030` nats tilted improvement) structurally unreachable in this regime. Two factors explain this:
   - At `600` steps from scratch (`val_ce ≈ 1.08`, perplexity `~2.95`), the model is still predominantly learning bekern syntax/rhythm and coarse visual layout rather than fine `3–5 px` staff-line vs space pitch discrimination where `±6°` tilt breaks axis-aligned strips.
   - Stock `AnisotropicMusicBlock` already pairs `1x9`/`9x1` strips with `3x3` local and `5x5` dilated branches plus 6 global self-attention layers, so `ctrl` is already largely insensitive to `±6°` tilt at this convergence level.
5. **Anomaly #3 — Missing QAT / `int4_dequant_val_ce`:** Unlike `i15-38`, `i15-11/run_colab.py` did not run QAT or disk-packed INT4 dequantization eval; `packed_int4_mb` (`53.194 MB`, `+0.155 MB`) is analytic from `param_summary()`. Given the `DISCARD` verdict on float CE, no INT4 follow-up is needed.
6. **Hardware / speed / VRAM:** Clean execution with zero OOM or cuDNN stalls. `sheared_aniso_rope` adds `+72.39 ms/step` (`+3.55%`) and `+422.3 MB` peak VRAM (`11,504.2 MB`) for the 3-angle masked depthwise strip convs + per-pixel softmax gates in `stage3`/`stage4`.

---

## Verdict: `DISCARD`

`sheared_aniso_rope` fails the `README.md` `PROMOTE` gate (`ema_val_ce: +0.00030`, `ema_val_tilted_ce: -0.00035` vs required `<= -0.030`), and even after accounting for the un-warmed `0.997` EMA lag, its raw CE delta (`-0.00572` upright / `-0.00609` tilted) is sub-`0.006` nats while costing **`+3.55%` training latency** and **`+422.3 MB` VRAM**.

**Falsified hypothesis / root cause:**
- Adding 3-angle gated `(3, 9)`/`(9, 3)` sheared depthwise strips to `stage3`/`stage4` does not pay for its `+3.55%` compute and `+422 MB` VRAM overhead — `AnisotropicMusicBlock`'s existing `3x3` + dilated `5x5` branches and global attention already absorb `±6°` tilt (`+0.00073` nats raw degradation on `ctrl`).
- Future short (600-step) proxy runners MUST NOT use un-warmed `EMA_DECAY = 0.997` (which leaves `16.5%` random init in `ema.shadow`; use `0.99` or warmup `min(decay, (1+step)/(10+step))`).

---

## Follow-Up Action / Successor Experiment

- **None queued (`DISCARD`).** Do not port `ShearedStripConv2d` into `uni_omr_v2.py`; keep stock `AnisotropicMusicBlock` and prioritize high-signal Iter-15 winners (e.g., `i15-38` `coordkey_ca`, `-0.02014` EMA / `-0.02635` raw, `-1.42%` faster).
- *Optional future note (low priority):* Because `PitchBiasedAnisoRoPE2D` has `0` extra params and `0` latency cost, if vertical pitch errors remain a top bottleneck after full convergence (`val_ce < 0.4` on `suite31_hw`), it can be re-evaluated in isolation on a converged/longer run rather than a 600-step from-scratch proxy.
