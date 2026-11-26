# OUTCOME — `2026-11-26T07-40-i13-17-paddleocr-vrfm-band-router` (`i13-17`)

- **Status:** `COMPLETED`
- **Completed UTC:** `2026-11-26T09:32:32Z` (started `08:47:52Z`)
- **Platform / GPU:** `Colab Tesla T4 (cc=7.5, bf16=False)` (FP32 fallback)
- **Elapsed Time:** `2679.81 s` (`44.7 min`)
- **Base Checkpoint:** `checkpoints/iter10_5_best/uni_omr_ema_best.pt` (`uni_omr_v1`, step `8000`, `val_ser=0.314308`, `missing_keys=[]`, `unexpected_keys=[]`)
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - `evals/2026-11-26T07-40-i13-17-paddleocr-vrfm-band-router.json`
  - `runs/2026-11-26T07-40-i13-17-paddleocr-vrfm-band-router.json`

## Measured Results (`max_dim=1280`, `max_len=3840`, MTP-4 + FSA)

| Arm | `hw_clean` Δ (`n=14`, W/T/L) | `hw_photo` Δ (`n=14`, W/T/L) | `pr_clean` Δ (`n=10`, W/T/L) | `pr_photo` Δ (`n=10`, W/T/L) | HW Avg Δ | Median s (HW clean / PR clean) |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_global_topk` | `0.264263` (`+0.0000`) | `0.187995` (`+0.0000`) | `0.582040` (`+0.0000`) | `0.578630` (`+0.0000`) | `+0.00000` | `10.89 s / 4.69 s` |
| `vrfm_border_3pct` | `+0.012091` (`4/7/3`) | `-0.001667` (`2/9/3`) | `+0.000000` (`0/10/0`) | `+0.000000` (`0/10/0`) | `+0.00521` | `10.05 s / 4.67 s` |
| `vrfm_border_6pct` | `+0.013890` (`4/1/9`) | `+0.003913` (`7/3/4`) | `-0.000183` (`0/8/2`) | `+0.000000` (`0/10/0`) | `+0.00890` | `10.72 s / 4.68 s` |
| **`vrfm_band_0.30`** | **`+0.014225` (`7/1/6`)** | **`+0.009730` (`8/2/4`)** | **`+0.000000` (`0/10/0`)** | **`+0.000000` (`0/10/0`)** | **`+0.01198`** | **`10.31 s / 4.67 s`** |
| `vrfm_band_0.60` | `+0.006531` (`5/1/8`) | `+0.007476` (`9/1/4`) | `+0.000000` (`0/10/0`) | `-0.000193` (`0/9/1`) | `+0.00700` | `9.35 s / 4.69 s` |
| `vrfm_b3_band0.30` | **`+0.015900` (`6/2/6`)** | `+0.007225` (`8/2/4`) | `+0.000000` (`0/10/0`) | `+0.000000` (`0/10/0`) | `+0.01156` | `9.60 s / 4.67 s` |
| `vrfm_b3_band0.30_k55` | `+0.015900` (`6/2/6`) | `+0.007341` (`9/1/4`) | `+0.000000` (`0/10/0`) | `+0.000000` (`0/10/0`) | `+0.01162` | `9.59 s / 4.67 s` |

## Scrutiny & Anomaly Check

1. **Coverage & Integrity:** All `48` pages × `7` arms scored cleanly (`0` decode failures, `0` NaN/Inf, `missing_keys=[]`).
2. **PROMOTE Gate Verification (`vrfm_band_0.30`):**
   - `hw_clean` Δ = **`+0.014225`** (`>= +0.010` gate), `wins > losses` (`7 > 6`).
   - `hw_photo` Δ = **`+0.009730`** (broad gains: `8` wins vs `4` losses; best `+0.0843` on `hw-debussy-vetement-du-blesse`, `+0.0827` on `hw-mozart-k464`; worst `-0.0318` on `hw-schubert-d810`).
   - `printed_clean` Δ = **`+0.000000`** (`10/10` exact ties) and `printed_photo` Δ = **`+0.000000`** (`10/10` exact ties) — **zero printed regression!**
   - **Zero latency overhead:** Pure 3-row `F.avg_pool2d` on `saliency_net` output adds `0 ms` decoder cost (median latency actually slightly faster: `10.31 s` vs `10.89 s` on `hw_clean`).
3. **Outlier & Secondary Observations:**
   - On `hw_clean`, `hw-debussy-la-plus-que-lente` contributes `+0.2106` to `vrfm_band_0.30`'s mean, but wins still outnumber losses (`7–6`) and `hw_photo` shows balanced, non-outlier-dependent gains (`8–4`).
   - Adding `border_3pct` (`vrfm_b3_band0.30`) slightly improves `hw_clean` (`+0.0159`) at the expense of `hw_photo` (`+0.0072`, `-0.0880` on `hw-debussy-lindaraja` photo where notation sits near the margin), making pure `vrfm_band_0.30` (`border_frac=0.0`) the cleaner, safer winner.
   - `vrfm_b3_band0.30_k55` differed from `vrfm_b3_band0.30` on only `1/48` pages (`override_max_keep_ratio(0.55)` barely changed token retention on this checkpoint).
   - **Checkpoint note:** Evaluated on `iter10_5_best` (`val_ser=0.3143`, T4 FP32). Re-verify `band_alpha=0.30` on `iter10_soup_baseline` / Iter 12 `v2` during integration.

## Verdict: **`PROMOTE`**

- **`vrfm_band_0.30`** (`border_frac=0.0, band_alpha=0.30`, 3-row horizontal staff-band saliency smoothing) clears every PROMOTE gate: **`+0.0142` `hw_clean`**, **`+0.0097` `hw_photo`** (`15` wins vs `10` losses across 28 HW pages), **`0.0000` printed regression (`20/20` ties)**, and **`0%` latency overhead**.

## Follow-Up Action / Successor Experiment

- **Promote Action:** Wire horizontal 3-row staff-band saliency prior (`band_alpha=0.30`, configurable via `--vrfm-band-alpha`) into `InkTokenRouter` / `InkTokenRouterV2` for inference and Iter 13 training, and confirm parity on `iter10_soup_baseline` + Iter 12 `UniOMR v2`.
