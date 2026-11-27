# OUTCOME — `2026-11-27T14-50-i15-46-xf-bd3lm-remdm-cell-repair` (`i15-46`)

- **Status:** `COMPLETED` (`rc=0`, `2/2` arms completed, `600/600` steps/arm, `muon+adamw`, zero NaNs, `error: null`)
- **Verdict:** **`PROMOTE`** *(preserves AR CE within `+0.00221 <= +0.005`, delivers `+59.06%` corrupted-span repair and `+10.94%` EMA / `+10.50%` INT4 absolute net accuracy gain on lowest-20%-confidence AR spans, and stays under the `55.0 MB` hard cap at `54.298 MB` exact disk bytes, `+1.259 MB` head overhead)*
- **Platform / GPU:** Local Windows 11 — `NVIDIA GeForce RTX 4080 SUPER` (16 GB VRAM, native BF16, `--no-cudnn` + CUTLASS `EFFICIENT_ATTENTION`)
- **Started / Finished (UTC):** `2026-11-27T18:29:12Z` → `2026-11-27T19:07:27Z` (`2293.11 s` / `38.2 min` total)
- **Baseline Commit SHA:** `6055c49`
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-27T14-50-i15-46-xf-bd3lm-remdm-cell-repair.json` (pulled to [`eval_hf.json`](eval_hf.json))
  - Run Ledger: `runs/2026-11-27T14-50-i15-46-xf-bd3lm-remdm-cell-repair.json` (pulled to [`run_hf.json`](run_hf.json))
  - Arm Syncs: `checkpoints/2026-11-27T14-50-i15-46-xf-bd3lm-remdm-cell-repair/{ctrl,bd3_remdm_head}_arm_result.json`

---

## Measured Results (`ctrl` vs `bd3_remdm_head`, 600 steps/arm, `muon+adamw`, `seed=42`)

| Metric | `ctrl` (`edge85m + ax0_gl6`) | `bd3_remdm_head` (`+ BD3ReMDMRepairHead`) | Delta (`Arm 1 - Arm 0`) | Gate (`PLAN.md`) | Pass? |
|---|---:|---:|---:|---|---|
| **`ema_val_ce`** *(AR primary)* | `1.12802` | `1.13023` | **`+0.00221`** | `<= +0.005` PROMOTE | **PASS** |
| **`int4_dequant_val_ce`** *(AR primary)* | `1.12919` | `1.13075` | **`+0.00156`** | `<= +0.005` | **PASS** |
| `raw_val_ce` | `1.10560` | `1.10871` | `+0.00311` | — | **PASS** |
| `int4_dequant - ema` gap | `+0.00117` | `+0.00052` | `-0.00065` | `<= +0.025` | **PASS** |
| **`remdm_corrupt_acc` (EMA: step0 → 1 → 2)** | N/A | `0.00187 → 0.57223 → 0.59251` | **`+59.06%` abs. gain** | `>= +15%` abs. PROMOTE | **PASS** |
| **`remdm_corrupt_acc` (INT4: step0 → 1 → 2)** | N/A | `0.00187 → 0.57141 → 0.59169` | **`+58.98%` abs. gain** | `>= +15%` abs. | **PASS** |
| **`remdm_ar_lowconf_acc` (EMA: before → after)** | N/A | `0.25784 → 0.36722` | **`+10.94%` abs. (`+42.4%` rel.)** | `> 0` net gain PROMOTE | **PASS** |
| **`remdm_ar_lowconf_acc` (INT4: before → after)** | N/A | `0.25068 → 0.35572` | **`+10.50%` abs. (`+41.9%` rel.)** | `> 0` net gain | **PASS** |
| **`packed_int4_mb`** *(exact disk bytes)* | `53.039 MB` | **`54.298 MB`** | **`+1.259 MB`** | `< 55.0 MB` & `< 1.5 MB` delta | **PASS** |
| `step_ms` *(training steady mean)* | `1,598.83 ms` | `1,986.52 ms` | `+387.69 ms` (`+24.2%`) | training-only overhead | **OK** |
| `train_elapsed_sec` | `963.07 s` | `1,190.70 s` | `+227.63 s` (`+23.6%`) | — | — |
| `peak_vram_mb` | `10,625.1 MB` | `11,380.0 MB` | `+754.9 MB` (`+7.1%`) | `< 14,000 MB` WDDM safe | **PASS** |
| **Overall Gate** | baseline | — | — | all PROMOTE gates | **PASS → `PROMOTE`** |

---

## Scrutiny & Anomaly Check

1. **Silent failures / partial coverage:** None. Both arms completed `600/600` steps with verified `optimizer_mode == "muon+adamw"`, `bf16 == true`, `seed == 42`, zero NaNs, `QATInt4ControllerV2` active from step `420` (covering both base model and `bd3_repair_head`), and exact on-disk Hybrid INT4 packing (`53,039,072 B` vs `54,297,568 B`).
2. **Zero target leakage in `remdm_ar_lowconf` repair evaluation (with proxy caveat):**
   - Verified `run_colab.py:384–405`: Task 2 passes `ar_pred` (the AR decoder's own argmax predictions, **never** `tgt_y`) with bottom-20%-confidence positions masked into `bd3_repair_head`. Step 2 remasks the bottom-50% confidence tokens from Step 1 and runs a second bidirectional pass.
   - **Proxy caveat:** `ar_logits` were computed in teacher-forced validation mode (`tgt[:, :-1]`), which is standard for 600-step CE proxies. Free-running autoregressive inference (where exposure bias can shift neighbor context) should be verified on `suite31_hw` once `BD3ReMDMRepairHead` is wired into Iter 15.
3. **INT4 quantization resilience:** Quantizing `bd3_repair_head` to group-32 INT4 caused almost zero repair degradation (`+59.06% → +58.98%` on corrupted spans, `+10.94% → +10.50%` on low-confidence AR spans), and the AR `int4_dequant - ema` gap was actually smaller (`+0.00052` vs `+0.00117` in `ctrl`).
4. **Benign `param_summary` vs `packed_int4_verification` accounting mismatch:** `param_summary.hybrid_fp16_int4_with_scales_mb` reported `57.501 MB` because the base `UniOMRv2Net.param_summary()` did not recognize the monkey-patched `bd3_repair_head` and counted its `2.23M` weights in `fp16_keep_params`. Actual on-disk serialization (`packed_int4_verification`) properly quantized all `17` Linear layers of `bd3_repair_head` to INT4 (`n_int4_tensors: 131 → 148`, `fp16_keep_mb: 12.855 → 12.860`), verifying **`54.298 MB < 55.0 MB`** (`+1.259 MB`). When porting into `uni_omr_v2.py`, register `bd3_repair_head` in `param_summary()`'s INT4 list so projected and disk sizes match.
5. **Training cost & budget headroom:**
   - `step_ms` rose `+24.2%` (`1599 → 1987 ms`) from running 2 complementary masked views every step; in full Iter 15 training, using 1 view/step (or enabling the repair loss only in the final 40% of steps with optional `mem_final.detach()`) can halve training overhead while eliminating the tiny `+0.00221` AR CE drag.
   - `54.298 MB` leaves `0.702 MB` headroom under the `55.0 MB` hard cap — if combined with `i15-45` QVLA mixed-bit INT8, cap QVLA `keep_frac` at `0.02` (`~+0.6 MB`) or use `d_rep=192` so the combined binary stays strictly `< 55.0 MB`.

---

## Verdict & Follow-Up Action / Successor Experiment

- **Verdict:** **`PROMOTE`** — clears every `PLAN.md` PROMOTE threshold (`ema_val_ce` delta `+0.00221 <= +0.005`, corrupted-span 2-step repair `+59.06% >= +15%`, low-confidence AR span net gain `+10.94%` EMA / `+10.50%` INT4, packed INT4 `54.298 MB < 55.0 MB` with `+1.259 MB < 1.5 MB` head overhead).
- **Follow-Up Action (for main thread batch promotion):**
  - Wire `BD3ReMDMRepairHead` (`block_size=32`, optional `--bd3-repair-weight 0.25`) into `tmp-train/mvd/uni_omr_v2.py` (including `param_summary()` INT4 accounting and `QATInt4ControllerV2` coverage), `train_uni_v2.py`, and optional 2-step post-AR low-confidence/violated-bar repair in `eval_uni_suite31.py`, with CPU unit tests.
