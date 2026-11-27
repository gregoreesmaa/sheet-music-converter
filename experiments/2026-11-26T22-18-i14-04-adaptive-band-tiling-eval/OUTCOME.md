# OUTCOME — `2026-11-26T22-18-i14-04-adaptive-band-tiling-eval` (`i14-04`)

- **Experiment ID:** `2026-11-26T22-18-i14-04-adaptive-band-tiling-eval`
- **Short ID / Wave:** `i14-04` (Wave 11 — Optical Resolution & Token-Cap Wall)
- **Status:** `COMPLETED (INVALIDATED BY HARNESS BUGS — ITERATE)`
- **Verdict:** **`ITERATE`** → Successor [`2026-11-27T00-52-i14-04b-adaptive-band-tiling-fixed`](../2026-11-27T00-52-i14-04b-adaptive-band-tiling-fixed/) (`i14-04b`)
- **Platform / GPU:** Kaggle `2× Tesla T4` (`cc=7.5`, single GPU used, `bf16=false` → silent FP32 fallback; kernel `gregoreesmaa/omr-i14-04`)
- **Checkpoint Requested:** `checkpoints/iter12_v2_best/uni_omr_v2_ema_best.pt` (**failed to load — `missing=349, unexpected=7`**)
- **Timestamps (UTC):** Started `2026-11-26T21:34:31Z` → Completed `2026-11-27T00:49:32Z`
- **Elapsed Time:** `11,700.33 s` (`3.25 h`; Arm A `3045.57 s`, Arm B `3023.66 s`, Arm C `5615.23 s`)
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-26T22-18-i14-04-adaptive-band-tiling-eval.json`
  - Run Ledger: `runs/2026-11-26T22-18-i14-04-adaptive-band-tiling-eval.json`
  - Local Log: `D:\tmp\overnight-queue-logs\2026-11-26T22-18-i14-04-adaptive-band-tiling-eval.kaggle.log`

---

## Measured Results (Invalidated — Unloaded Random-Init Model)

All 90 `suite31_hw` pages (`31 printed + 14 HW` × `{clean, photo}`) were evaluated across all 3 arms, but **all Note F1 and `dec_ok` scores are identically `0.0000`** because the model ran with 100% randomly initialized weights (`missing=349, unexpected=7`). Router `ink_recall` values also reflect a randomly initialized router with default `max_mem_tokens=10240` rather than Iter 12's trained router (`max_mem_tokens=3072`).

| Slice | N | Arm A (`1408px`) F1 / `dec_ok` / `ink_rec` / `lat_s` | Arm B (`1792px`) F1 / `dec_ok` / `ink_rec` / `lat_s` | Arm C (Band-Tiling) F1 / `dec_ok` / `ink_rec` / `lat_s` | Δ (C − A) F1 | Δ (B − A) F1 |
|---|---:|---|---|---|---|---|
| **`all90`** (`all45` clean+photo) | 90 | `0.0000` / `0.00` / `0.9767` / `33.69s` | `0.0000` / `0.00` / `0.9685` / `33.45s` | `0.0000` / `0.00` / `0.9746` / `62.25s` | `0.0000` | `0.0000` |
| **`dense_subset`** | 85 | `0.0000` / `0.00` / `0.9753` / `33.71s` | `0.0000` / `0.00` / `0.9667` / `33.45s` | `0.0000` / `0.00` / `0.9731` / `63.95s` | `0.0000` | `0.0000` |
| **`sparse_subset`** | 5 | `0.0000` / `0.00` / `0.9993` / `33.39s` | `0.0000` / `0.00` / `0.9999` / `33.48s` | `0.0000` / `0.00` / `0.9993` / `33.42s` | `0.0000` | `0.0000` |
| **`printed_clean`** | 31 | `0.0000` / `0.00` / `0.9971` / `33.89s` | `0.0000` / `0.00` / `0.9978` / `33.49s` | `0.0000` / `0.00` / `0.9980` / `63.53s` | `0.0000` | `0.0000` |
| **`printed_photo`** | 31 | `0.0000` / `0.00` / `0.9999` / `33.85s` | `0.0000` / `0.00` / `0.9999` / `33.60s` | `0.0000` / `0.00` / `1.0000` / `65.49s` | `0.0000` | `0.0000` |
| **`hw_clean`** | 14 | `0.0000` / `0.00` / `0.8790` / `33.33s` | `0.0000` / `0.00` / `0.8347` / `33.29s` | `0.0000` / `0.00` / `0.8699` / `77.95s` | `0.0000` | `0.0000` |
| **`hw_photo`** | 14 | `0.0000` / `0.00` / `0.9776` / `33.26s` | `0.0000` / `0.00` / `0.9679` / `33.19s` | `0.0000` / `0.00` / `0.9709` / `36.56s` | `0.0000` | `0.0000` |

---

## Scrutiny & Anomaly Check

1. **Catastrophic Silent Checkpoint Load Failure (`missing=349, unexpected=7`):**
   - Line 9 of `2026-11-26T22-18-i14-04-adaptive-band-tiling-eval.kaggle.log` reads:
     ```text
     Loaded frozen ckpt (checkpoints__iter12_v2_best__uni_omr_v2_ema_best.pt): d_model=384 max_mem=10240 missing=349 unexpected=7
     ```
   - **Root cause:** `checkpoints/iter12_v2_best/uni_omr_v2_ema_best.pt` stores its state dict under `ck["model"]` and its config under `ck["config"]` (the canonical format handled by `eval_uni_suite31.load_uni_checkpoint`). `run_colab.py`'s custom `_load_frozen_model` checked only `raw.get("ema_state_dict") or raw.get("model_state_dict") or raw.get("state_dict") or raw` and `raw.get("cfg") or raw.get("model_cfg") or {}`. Both missed, so `state` fell through to the 7-key top-level dict (`unexpected=7`), `cfg` fell back to default `small_45m()` (`max_mem=10240`), and `load_state_dict(state, strict=False)` silently swallowed all **349 missing model parameters**.
   - **Consequence:** Every single decode step ran a 100% randomly initialized model, never emitted `<eos>`, hit `max_len=2560` on every page, and produced `dec_ok=0.0` / `note_f1=0.0` across all 90 pages × 3 arms.

2. **Secondary Bug — Wrong Input Normalization Range (`[-1, 1]` vs `[0, 1]`):**
   - `run_colab.py`'s `_pil_to_tensor` returned `(t - 0.5) / 0.5` (`[-1, 1]`), whereas `uni_omr._pil_to_chw_tensor` feeds `[0.0, 1.0]` (`mul_(1.0 / 255.0)`). Even with weights loaded, `[-1, 1]` would severely degrade the frozen vision encoder.

3. **Hardware / Platform Quirk — Silent FP32 Fallback on Kaggle T4 (`3.25 h` runtime):**
   - `use_bf16 = (cc[0] >= 8)` evaluated to `False` on Tesla T4 (`cc=7.5`) with no `torch.float16` autocast branch, forcing pure FP32 inference (`33.5 s/page` hitting `max_len=2560`) and burning `3.25 h` of Kaggle GPU quota without an early fail-fast sanity check.

4. **Density Classifier Calibration Over-Trigger (`85/90` pages = `94.4%` marked dense):**
   - In `per_page_sample`, `piece_0..piece_7` reported `staff_space_px_1408 = 2.79–2.99 px` because `min_lag = max(3, int(round(H * 0.002)))` latched onto a `3 px` line-thickness/noise lag instead of the true staff-space period (typically `7–12 px` at `1408px`). As a result, `85/90` pages triggered tiling instead of only genuinely dense pages.

5. **Coverage & Sanity:** All 90 pages (`31+31+14+14`) completed with `rc=0` and zero NaNs, and Kaggle kernel `gregoreesmaa/omr-i14-04` has terminated cleanly (no idle GPU burn).

---

## Verdict: `ITERATE`

The adaptive band-tiling hypothesis **was not tested** due to the harness checkpoint-loading and input-normalization bugs. We **MUST NOT** discard or promote based on `0.0000` random-init scores; instead we fix all four bugs immediately and re-run as **`i14-04b`**.

## Follow-Up Action / Successor Experiment

- **Queued Successor:** [`2026-11-27T00-52-i14-04b-adaptive-band-tiling-fixed`](../2026-11-27T00-52-i14-04b-adaptive-band-tiling-fixed/) (`i14-04b`, registered in `experiments/overnight_queue/queue.json` with `depends_on: ["2026-11-26T22-18-i14-04-adaptive-band-tiling-eval"]`, `provider: "colab_l4"`).
- **Fixes in `i14-04b`:**
  1. Load frozen checkpoint via canonical `eval_uni_suite31.load_uni_checkpoint(ckpt_local, device)` with a **hard `assert len(meta["missing_keys"]) == 0`**.
  2. Preprocess images via canonical `uni_omr._resize_fullpage_pil` + `uni_omr._pil_to_chw_tensor` (`[0, 1]` range).
  3. Enable native `bfloat16` on `cc >= 8` (L4/A100) and `float16` autocast on `cc < 8` (T4) — never silent FP32.
  4. Fix `staff_space_px` autocorrelation estimator (`min_lag >= max(6, int(round(H * 0.0035)))` + local-maximum peak selection) so sparse vs dense separation is physically accurate.
  5. Add a **2-page fail-fast sanity gate** on Arm A (`assert mean_dec_ok > 0.0 and mean_note_f1 > 0.05` after first 2 printed clean pages) so any future harness regression aborts in `< 30 s` instead of burning hours.
