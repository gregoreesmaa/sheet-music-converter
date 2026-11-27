# OUTCOME — `2026-11-27T10-30-i15-37-modegate-dualhead-text-and-chained-mtp`

- **Status:** `COMPLETED` (**CONFOUNDED** — Arm 1 trained with `opt=adamw` instead of `opt=muon+adamw` due to `"base."` wrapper prefix bug, fixed in `95446af` / `b64fd64`)
- **Verdict:** **`ITERATE`** (CONFOUNDED POSITIVE — strong signal on `ema_val_ce`, `val_ce`, and `spec_accept_len`, requires clean matched `muon+adamw` + INT4-QAT verification in `i15-37b`)
- **Short ID / Wave:** `i15-37` (Iter 15 — Weight-Incompatible Architecture Proxy, `wave=0`)
- **Started / Finished (UTC):** `2026-11-27T11:40:53Z` → `2026-11-27T12:23:20Z` (`2547.20 s` / `42.5 min` total wall-clock)
- **Platform / GPU:** Local `NVIDIA GeForce RTX 4080 SUPER` (`16 GB` VRAM, native BF16; executed via `D:\tmp\run_local_p0_queue.py` / `task-4713`, `seed=42`, `600` steps/arm)
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-27T10-30-i15-37-modegate-dualhead-text-and-chained-mtp.json` (mirrored locally at [eval_hf.json](./eval_hf.json))
  - Run Ledger: `runs/2026-11-27T10-30-i15-37-modegate-dualhead-text-and-chained-mtp.json` (mirrored locally at [run_hf.json](./run_hf.json))

---

## Measured Results (`ctrl` vs `modegate_cmtp`, 600 steps/arm, `iter10_5_proxy_2k`)

| Metric | `ctrl` (`opt=muon+adamw`) | `modegate_cmtp` (`opt=adamw` — CONFOUNDED) | Delta (`modegate_cmtp - ctrl`) | `README.md` Gate | Status |
|---|---:|---:|---:|---|---|
| **Optimizer (actual)** | `muon+adamw` | **`adamw`** (prefix bug) | **MISMATCH** | matched `muon+adamw` | **CONFOUNDED** |
| **`ema_val_ce`** (primary, `decay=0.997`) | `1.29243` | **`1.19205`** | **`-0.10038`** (`-7.77%`) | `<= +0.015` (ideally `< 0`) | PASS (confounded) |
| **`val_ce`** (raw) | `1.08365` | **`1.03970`** | **`-0.04395`** (`-4.06%`) | — | PASS (confounded) |
| **`int4_dequant_val_ce`** | *not measured* | *not measured* | — | — | Added in `i15-37b` |
| **`final_25step_loss`** | `1.7063` | **`1.4937`** | `-0.2126` | — | PASS (confounded) |
| **`mtp_k1_acc`** (EMA / raw) | `0.55187` / `0.62105` | **`0.58516`** / **`0.63980`** | `+3.33 pp` / `+1.88 pp` | — | PASS (confounded) |
| **`mtp_k2_acc`** (EMA / raw, greedy-chain) | `0.42831` / `0.50375` | **`0.45459`** / **`0.50824`** | `+2.63 pp` / `+0.45 pp` | `>= +3.0 pp` (or `spec_accept_len >= +0.10`) | Marginal on raw `k2` |
| **`mtp_k3_acc`** (EMA / raw, greedy-chain) | `0.34668` / `0.42169` | **`0.36616`** / `0.41831` | `+1.95 pp` / `-0.34 pp` | — | Mixed on raw `k3` |
| **`mtp_k4_acc`** (EMA / raw, greedy-chain) | `0.26868` / `0.33705` | **`0.29098`** / **`0.33798`** | `+2.23 pp` / `+0.09 pp` | — | Flat on raw `k4` |
| **`mtp_tf_k2..k4_acc`** (EMA / raw, teacher-forced) | — | `0.5402 / 0.5482 / 0.5328` (raw `0.5928 / 0.5898 / 0.5743`) | — | diagnostic | Strong TF ceiling |
| **`spec_accept_len`** (EMA / raw) | `0.8838` / `1.1603` | **`1.1319`** / **`1.3443`** | **`+0.2481`** (`+28.1%`) / **`+0.1840`** (`+15.9%`) | `>= +0.10` | **PASS** (`2.48×` gate) |
| **`step_ms`** | `2073.18 ms` | `1945.28 ms` | `-127.90 ms` (`-6.17%`) | `<= +10%` | Confounded (no Muon NS) |
| **`elapsed_sec`** | `1243.91 s` | `1167.17 s` | `-76.74 s` | — | OK |
| **`peak_vram_mb`** | `11,075.5 MB` | `11,426.7 MB` | `+351.2 MB` | `< 14,000 MB` | **PASS** |
| **`packed_int4_stripped_mb`** | `53.039 MB` (`77.87M`) | `53.159 MB` (`78.07M`) | `+0.120 MB` (`+0.20M`) | `< 55.0 MB` | **PASS** |
| **`packed_int4_with_mtp_mb`** | `53.485 MB` (`78.65M`) | **`54.050 MB`** (`79.65M`) | `+0.565 MB` (`+0.99M`) | `<= 55.0 MB` | **PASS** (`0.950 MB` headroom) |

- **`gate_pass`:** Numerically clears every `README.md` threshold (`ema_val_ce -0.10038`, `spec_accept_len +0.2481` EMA / `+0.1840` raw, `packed_int4_with_mtp_mb 54.050 <= 55.0 MB`), **but invalid for direct `PROMOTE` due to the optimizer confound**.

---

## Scrutiny & Anomaly Check

1. **Silent failures / coverage:** Both arms completed `600/600` steps with `error: null`, `val_tokens: 53110` on both arms, zero NaNs, and both `eval.json` and `run.json` uploaded cleanly to HF Private Hub (`task-4713.log` lines 35, 63–65).
2. **CRITICAL CONFOUND — `"base."` wrapper prefix triggered AdamW fallback on Arm 1:**
   - `ModeGateChainedMTPWrapper` wraps `UniOMRv2Net` as `self.base`, prefixing all transformer parameter names with `"base."`.
   - Prior to commit `95446af` / `b64fd64` (`tmp-train/mvd/train_uni_v2.py`), `partition_muon_adamw_params()` matched `_MUON_TRANSFORMER_PREFIXES` without stripping `"base."`, causing Arm 1 (`modegate_cmtp`) to silently fall back to `opt=adamw` (`task-4713.log` line 37) while Arm 0 (`ctrl`) ran `opt=muon+adamw` (`task-4713.log` line 9).
   - **Why this confound matters in BOTH directions:** At `lr=3e-4` in a short 600-step from-scratch run, `AdamW` converged much faster in steps 1–75 (`step 25 ce=2.3328` vs `ctrl`'s `3.9705`). Because `i15-37` used `GPU_EMA_DECAY=0.997` (a ~333-step half-life on a 600-step run), early-step convergence heavily biased `ema_val_ce` (`-0.10038` nats). Skipping Muon's Newton-Schulz orthogonalization also artificially shaved `-127.9 ms/step` (`-6.17%`) off Arm 1's step time.
3. **Genuine architectural signal worth verifying:**
   - On `raw_eval` (step 600), marginal per-depth greedy accuracies at `k=2..4` were nearly identical (`+0.45 / -0.34 / +0.09 pp`), yet **joint prefix acceptance `spec_accept_len` jumped `+0.1840` (`1.3443` vs `1.1603`, `+15.9%`)** (and `+0.2481` / `+28.1%` on EMA). Conditioning depth `k` on the depth `k-1` draft token genuinely improves joint prefix coherence over independent parallel MTP heads, and the total footprint stays safely under the 55 MB WebGPU budget (`54.050 MB` with MTP, `53.159 MB` stripped).
4. **Missing INT4-QAT check:** Unlike `i15-38`, `i15-33`, and `i15-14`, `i15-37` did not enable `QATInt4ControllerV2` or measure `int4_dequant_val_ce`. Added in `i15-37b` alongside `EMA_DECAY=0.99` and `assert opt.mode == "muon+adamw"`.

---

## Verdict: `ITERATE`

Mark **`ITERATE`** (CONFOUNDED POSITIVE). Do not promote `modegate_cmtp` into `uni_omr_v2.py` until it beats `ctrl` cleanly under matched `opt=muon+adamw` on both arms with `QATInt4ControllerV2` and `int4_dequant_val_ce` verification.

---

## Follow-Up Action / Successor Experiment

- **Queued Successor (`wave=0`, `LOCAL_ELIGIBLE`):** [`2026-11-27T14-30-i15-37b-modegate-cmtp-muon-clean`](../2026-11-27T14-30-i15-37b-modegate-cmtp-muon-clean/PLAN.md)
  - Hard-asserts `opt.mode == "muon+adamw"` on both `ctrl` and `modegate_cmtp` (enabled by the `"base."` prefix strip fix in `95446af` / `b64fd64`).
  - Matches `i15-38`/`i15-33`/`i15-14` protocol: `EMA_DECAY=0.99`, `QATInt4ControllerV2` at step `420`, exact disk-verified Hybrid FP16/INT4 `int4_dequant_val_ce`, and HF partial arm auto-resume.
