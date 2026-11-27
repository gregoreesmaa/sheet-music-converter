# OUTCOME — `2026-11-27T10-30-i15-40-pro200m-d768-capacity-proxy` (`i15-40`)

- **Status:** `COMPLETED` (`2 / 2` arms completed, `rc=0`)
- **Verdict:** **`ITERATE`**
- **Short ID / Wave:** `i15-40` (Wave 0 / Wave 13 — Weight-Incompatible PRE-ITER-15 Architecture Proxy)
- **Started / Finished (UTC):** `2026-11-27T13:35:58Z` → `2026-11-27T14:07:40Z` (`1,901.72 s` / `31.7 min` GPU compute; `3,995.11 s` queue wall-clock incl. A100 provisioning retries)
- **Platform / GPU:** Google Colab `NVIDIA A100-SXM4-40GB` (native BF16, `600` steps/arm, `seed=42`, `grad_accum=4`, `AdamW`, `lr=3e-4`, `weight_decay=0.04`, WSD warmup=`60` / decay=`25%`, `EMA=0.99`, `QATInt4ControllerV2` at `step >= 420`)
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-27T10-30-i15-40-pro200m-d768-capacity-proxy.json` (mirrored locally at [eval_hf.json](./eval_hf.json))
  - Run Ledger: `runs/2026-11-27T10-30-i15-40-pro200m-d768-capacity-proxy.json` (mirrored locally at [run_hf.json](./run_hf.json))

---

## Measured Results (`edge85m_ctrl` vs `pro200m`, 600 steps/arm, `iter10_5_proxy_2k`)

| Metric | `edge85m_ctrl` (`d=512`, `0ax+6gl`, `10L` dec) | `pro200m` (`d=768`, `0ax+8gl`, `12L` dec) | Delta (`pro200m - edge85m_ctrl`) | `PLAN.md` Gate | Status |
|---|---:|---:|---:|---|---|
| **`ema_val_ce`** (primary) | `1.19315` | **`1.16625`** | **`-0.02690`** (`-2.25%`) | `PROMOTE <= -0.04` / `ITERATE -0.02..-0.04` | **ITERATE** |
| **`raw_val_ce`** | `1.17620` | **`1.15034`** | **`-0.02586`** (`-2.20%`) | Diagnostic | Improved |
| **`int4_dequant_val_ce`** | `1.19312` | **`1.16626`** | **`-0.02686`** (`-2.25%`) | Diagnostic | Improved |
| **INT4 quant drop** (`int4 - ema`) | `-0.00003` | `+0.00001` | `+0.00004` | `<= +0.015` | **PASS** (lossless) |
| **`train_loss_tail50`** | `1.61619` | **`1.57690`** | **`-0.03929`** (`-2.43%`) | Diagnostic | Improved |
| **`step_ms`** (steady train avg) | `1,368.41 ms` | `1,458.44 ms` | `+90.03 ms` (`+6.58%`, **`1.066×`**) | Throughput | **PASS** (`+6.6%` for `2.47×` params!) |
| **`elapsed_secs`** (arm train) | `885.36 s` | `939.14 s` | `+53.78 s` (`+6.07%`) | — | Minimal overhead |
| **`peak_vram_mb`** (`vram_gb`) | `6,324.2 MiB` (`6.176 GB`) | `10,933.2 MiB` (`10.677 GB`) | `+4,609.0 MiB` (`+4.501 GB`, `+72.88%`) | `< 14 GB` local WDDM / `40 GB` A100 | **PASS** (fits RTX 4080S 16 GB) |
| **`packed_int4_mb`** (disk-verified) | `53.039 MB` (`53,039,072 B`) | **`119.156 MB`** (`119,155,952 B`) | `+66.117 MB` (`2.247×`) | **`<= 135.0 MB`** Pro budget | **PASS** (`15.844 MB` headroom) |
| `int4_w` / `scales` / `fp16_keep` | `35.719` / `4.465` / `12.855 MB` | `92.114` / `11.514` / `15.527 MB` | `+56.395` / `+7.049` / `+2.672 MB` | Exact binary match | **PASS** (`0 B` discrepancy) |
| **`infer_params`** (`total_params`) | `77,866,208` (`79,224,715`) | `191,992,288` (`194,827,147`) | `+114,126,080` (`2.466×` infer) | `~185–205M` target | **PASS** |
| **`gate_pass`** (`PLAN.md` `PROMOTE`) | — | **FALSE** (`ITERATE` band) | — | All `PLAN.md` `PROMOTE` gates | **ITERATE** |

---

## Scrutiny & Anomaly Check

1. **Silent failures / coverage & reproducibility:** None. Both arms completed all `600/600` steps (`status: "completed"`), zero NaNs, `2,000` train / `64` val loaded cleanly on Colab `NVIDIA A100-SXM4-40GB`, and the Colab session (`2026-11-27t10-30-i15-40-`) terminated cleanly (`exit_code=0`). `edge85m_ctrl` closely reproduced sibling Wave-0 `AdamW` controls (`i15-36` `1.19388`, `i15-39` `1.19410`) at `ema_val_ce = 1.19315` (`train_loss_tail50 = 1.61619` vs `1.61609`), confirming an uncorrupted baseline.
2. **Exact size & lossless INT4 quantization:** `param_summary.hybrid_fp16_int4_with_scales_mb` matches the disk-verified binary byte-for-byte (`53,039,072 B` = `53.039 MB` for `edge85m_ctrl`; `119,155,952 B` = `119.156 MB` for `pro200m`). `pro200m` sits `15.844 MB` under the `135.0 MB` Pro-tier budget, and `QATInt4ControllerV2` (`group_size=32`, enabled at `step=420`) incurs virtually zero quantization loss (`int4_dequant_val_ce - ema_val_ce = +0.00001`).
3. **Standout throughput & memory profile:** Scaling inference parameters by **`2.47×`** (`77.87M → 191.99M`) added only **`+6.58%` step latency** (`1,368.41 → 1,458.44 ms/step`, `1.066×`) on A100 because `AnisotropicMusicBackbone` + PIL data loading dominate step time at `batch_size=1 × grad_accum=4`. Peak VRAM is only **`10.677 GB`** with `grad_checkpoint=True`, meaning `pro200m` even trains comfortably inside local RTX 4080 SUPER (`16 GB`, `< 14 GB` WDDM budget).
4. **Why `pro200m` landed at `-0.02690` `ema_val_ce` (inside the `ITERATE` band `[-0.04, -0.02]`, short of the `-0.04` `PROMOTE` gate):**
   - **Short 600-step horizon under-trains `192M` params:** At `600 × 4 = 2,400` page updates (~1.2 epochs) from scratch, `pro200m`'s training-loss lead over `edge85m_ctrl` was still accelerating into the WSD tail: `-0.0130` at step 400 (`1.8173 vs 1.8303`) → `-0.0296` at step 500 (`1.7629 vs 1.7925`) → **`-0.0393` at step 600** (`1.5769 vs 1.6162`).
   - **Unscaled `AdamW` LR (`3e-4`) instead of `muon+adamw` / $\mu$P-scaled LR (`2e-4`):** `run_colab.py` line 397 used plain `torch.optim.AdamW` at the same `lr=3e-4` for both `d=512` and `d=768`. Wider `d=768` models typically benefit from `lr=2e-4` (or `HybridOptimizerV2` `muon+adamw`) and a slightly longer horizon (`1,200` steps) to unlock the full `-0.04+` scaling gap.
   - **Symmetric EMA lag:** `EMA_DECAY=0.99` over 600 steps lags `raw_val_ce` by `~0.016` on both arms (`1.16625 vs 1.15034` for `pro200m`; `1.19315 vs 1.17620` for `edge85m_ctrl`).

---

## Verdict: **`ITERATE`**

- **Why `ITERATE` (exact `PLAN.md` Section 6 match):** `pro200m` cleared both hard constraints with large margins (`packed_int4_mb = 119.156 MB <= 135.0 MB`, `int4_drop = +0.00001 <= 0.015`, `step_ms_ratio = 1.066×`, `vram_gb = 10.677 GB`) and beat `edge85m_ctrl` cleanly on every loss metric (`ema_val_ce -0.02690`, `raw_val_ce -0.02586`, `int4_dequant_val_ce -0.02686`, `train_loss_tail50 -0.03929`), landing squarely in the `ITERATE` band (`-0.02` to `-0.04`) rather than the `<= -0.04` `PROMOTE` gate.

---

## Follow-Up Action / Successor Experiment

- **Recommended Successor (`i15-40b-pro200m-muon-lr2e4-1200step`):** Re-test `pro200m` (`d=768`, `192M`, `119.16 MB` INT4) vs `edge85m_ctrl` (`d=512`, `77.87M`, `53.04 MB` INT4) with:
  1. Matched **`HybridOptimizerV2` (`muon+adamw`)** on both arms,
  2. $\mu$P-appropriate LR (`lr=2e-4` for `pro200m` vs `3e-4` for `edge85m_ctrl`, plus a `pro200m` `lr=3e-4` arm if budget allows), and
  3. **`1,200` steps/arm** (where the widening step `400→600` trajectory `-0.013 → -0.039` can mature past the `-0.04` `PROMOTE` threshold).
- **Iter 15 decision:** Keep `edge85m + ax0_gl6` (`53.04 MB` INT4) as the Iter 15 default WebGPU tier; hold `UniOMRv2Config.pro200m()` promotion as our Pro-tier / KD-teacher architecture pending the `i15-40b` `muon+adamw` + `1,200`-step verification.
