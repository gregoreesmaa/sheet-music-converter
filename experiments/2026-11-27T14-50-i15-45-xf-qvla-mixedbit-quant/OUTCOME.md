# OUTCOME — `2026-11-27T14-50-i15-45-xf-qvla-mixedbit-quant` (`i15-45`)

- **Status:** `COMPLETED` (`rc=0`, `2/2` arms completed, `64/64` held-out val pages, zero NaNs, `error: null`)
- **Verdict:** **`PROMOTE`** *(recovers `48.15%` of total INT4 CE degradation and `71.77%` of structural/header/pitch CE degradation for only `+0.815 MB` exact disk overhead and `11.1 s` PTQ calibration)*
- **Platform / GPU:** Local Windows 11 — `NVIDIA GeForce RTX 4080 SUPER` (16 GB VRAM, native BF16, `--no-cudnn` + CUTLASS `EFFICIENT_ATTENTION`)
- **Started / Finished (UTC):** `2026-11-27T18:26:53Z` → `2026-11-27T18:29:12Z` (`138.68 s` / `2.3 min` total)
- **Checkpoint Evaluated:** `hf:checkpoints/iter14-v2a-mid/uni_omr_v2_ema_latest.pt (step=6874)`
- **Baseline Commit SHA:** `6055c49`
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-27T14-50-i15-45-xf-qvla-mixedbit-quant.json` (pulled to [`eval_hf.json`](eval_hf.json))
  - Run Ledger: `runs/2026-11-27T14-50-i15-45-xf-qvla-mixedbit-quant.json` (pulled to [`run_hf.json`](run_hf.json))
  - Arm Syncs: `checkpoints/2026-11-27T14-50-i15-45-xf-qvla-mixedbit-quant/{uniform_int4,qvla_mixed_int4}_arm_result.json`

---

## Measured Results (`uniform_int4` vs `qvla_mixed_int4`, `iter14-v2a-mid` step 6874, 32 calib / 64 val pages, `seed=42`)

| Metric | FP16 Teacher | `uniform_int4` (Arm 0) | `qvla_mixed_int4` (Arm 1) | Delta (`Arm 1 - Arm 0`) | Gate (`PLAN.md`) | Pass? |
|---|---:|---:|---:|---:|---|---|
| **`int4_dequant_val_ce`** *(primary)* | `0.47574` | `0.48143` | **`0.47869`** | **`-0.00274`** | lower is better | **PASS** |
| **`dequant_ce_delta`** *(int4 − fp16)* | `0.00000` | `+0.00569` | **`+0.00295`** | **`-0.00274` (`48.15%` recovery)** | `>= 30%` recovery PROMOTE | **PASS** |
| **`struct_pitch_val_ce`** *(5× critical toks)* | `0.53947` | `0.54751` | **`0.54174`** | **`-0.00577`** | lower is better | **PASS** |
| **`struct_pitch_ce_delta`** *(int4 − fp16)* | `0.00000` | `+0.00804` | **`+0.00227`** | **`-0.00577` (`71.77%` recovery!)** | lower is better | **PASS** |
| **`packed_int4_mb`** *(exact disk bytes)* | — | `31.283 MB` | **`32.098 MB`** | **`+0.815 MB`** | `< 55.0 MB` & `<= +3.5 MB` | **PASS** |
| `int4_weight_mb` / `int4_scales_mb` | — | `17.303 / 2.163 MB` | `16.391 / 2.049 MB` | `-0.912 / -0.114 MB` | — | — |
| `fp16_keep_mb` *(incl. 5% INT8 channels)* | — | `11.817 MB` | `13.649 MB` | `+1.832 MB` | — | — |
| `mask_overhead_mb` / `extra_scale_bytes` | — | `0.0000 MB / 0 B` | `0.0094 MB / 256 B` | `+0.0094 MB / +256 B` | negligible | **PASS** |
| `quantvla_calibration` (`128` heads, 40 steps) | — | N/A | KL `0.01436 → 0.00026` (`-98.2%`) | scales `[0.8315, 1.1689]` | stable | **PASS** |
| `elapsed_sec` *(PTQ + calib + eval)* | — | `3.11 s` | `11.14 s` | `+8.03 s` | fast PTQ | **PASS** |
| **Overall Gate** | teacher | baseline | — | — | all PROMOTE gates | **PASS → `PROMOTE`** |

---

## Scrutiny & Anomaly Check

1. **Silent failures / partial coverage:** None. Both arms evaluated the exact same 64 held-out `proxy_val.jsonl` pages against the identical `iter14-v2a-mid` FP16 teacher (`fp16_val_ce = 0.47574`, `fp16_struct_pitch_val_ce = 0.53947`) with zero NaNs and exact on-disk binary serialization (`31,282,776 B` vs `32,098,146 B`).
2. **Exact bit-width verification (INT8 vs FP16 naming):** The JSON summary field is named `fp16_keep_mb` (`11.817 → 13.649 MB`, `+1.832 MB`) and `fp16_channel_frac_actual = 0.05`, while the actual binary packer (`run_colab.py:526–538`, matching `PLAN.md:23`) quantized the top-5% sensitive output channels to **symmetric per-channel INT8 + FP16 scale** (not FP16). This is strictly better for the `< 55 MB` edge budget: net disk growth is only **`+0.815 MB`** (`+2.6%`) while recovering **`48.15%`** of overall INT4 CE degradation and **`71.77%`** of critical structural/header/pitch CE degradation.
3. **Checkpoint scale vs `edge85m` budget headroom:** `iter14-v2a-mid` packed to `31.283 MB` uniform (`32.098 MB` QVLA) because it is smaller than `edge85m` (`53.039 MB` uniform INT4). On `edge85m`, 5% per-channel INT8 adds `~+1.5–1.7 MB` (`~54.6 MB`, verified `< 55.0 MB` in `--smoke-test`), which passes standalone. When combining QVLA mixed-bit with extra heads (e.g. `i15-46` `BD3ReMDMRepairHead` at `+1.259 MB`), expose `--qvla-int8-frac` (default `0.05`, or `0.02` when combined with repair head) so total packed size stays strictly `< 55.0 MB`.
4. **QuantVLA attention head scales (`128` heads, `+256 B`):** Calibration converged cleanly (`init_kl = 0.01436 → final_kl = 0.00026`, `-98.2%`, scales in `[0.8315, 1.1689]`, zero boundary saturation against `[0.7, 1.4]`). Because `q_norm` is `RMSNorm` (scale-invariant on its input), per-head scales cannot fold into `q_proj`; they should be folded into per-head `q_norm.weight` (or a `+256 B` post-`q_norm` vector) during ONNX/binary export.

---

## Verdict & Follow-Up Action / Successor Experiment

- **Verdict:** **`PROMOTE`** — cleanly passes every `PLAN.md` PROMOTE gate (`48.15% >= 30%` overall degradation recovery, `71.77%` structural/pitch recovery, `+0.815 MB <= +3.5 MB` and `32.098 MB < 55.0 MB` hard cap, `11.1 s` total PTQ calibration time).
- **Follow-Up Action (for main thread batch promotion):**
  - Wire QVLA 5×-weighted Taylor channel sensitivity (`--qvla-int8-frac 0.05`, configurable down to `0.02` if combined with `BD3ReMDMRepairHead`) + QuantVLA per-head `q_norm` attention temperature calibration (`40` steps KL distillation) into `tmp-train/mvd/export_uni_v2_onnx.py` and the Iter 15 post-training INT4 export pipeline, with CPU unit tests verifying packed binary size `< 55.0 MB` and dequant parity.
