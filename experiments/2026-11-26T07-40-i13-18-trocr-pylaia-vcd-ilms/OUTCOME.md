# OUTCOME — `2026-11-26T07-40-i13-18-trocr-pylaia-vcd-ilms` (`i13-18`)

- **Status:** `COMPLETED`
- **Completed UTC:** `2026-11-26T11:13:06Z` (started `08:28:06Z`)
- **Platform / GPU:** `Colab Tesla T4 (cc=7.5, bf16=False)` (FP32 fallback)
- **Elapsed Time:** `9899.59 s` (`165.0 min` / `2.75 h`)
- **Base Checkpoint:** `checkpoints/iter10_5_best/uni_omr_ema_best.pt` (`uni_omr_v1`, step `8000`, `val_ser=0.314308`, `missing_keys=[]`, `unexpected_keys=[]`)
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - `evals/2026-11-26T07-40-i13-18-trocr-pylaia-vcd-ilms.json`
  - `runs/2026-11-26T07-40-i13-18-trocr-pylaia-vcd-ilms.json`

## Measured Results (`max_dim=1280`, `max_len=3840`, `beta_plaus=0.10`, KV-cached FSA)

| Arm | `hw_clean` Δ (`n=14`, W/T/L) | `hw_photo` Δ (`n=14`, W/T/L) | `pr_clean` Δ (`n=10`, W/T/L) | `pr_photo` Δ (`n=10`, W/T/L) | HW Avg Δ | Median s (HW photo / PR clean) |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_alpha0.0` | `0.264263` (`+0.0000`) | `0.187995` (`+0.0000`) | `0.582040` (`+0.0000`) | `0.578630` (`+0.0000`) | `+0.00000` | `26.85 s / 12.93 s` |
| `vcd_blur2_a0.15` | `-0.019677` (`3/1/10`) | `+0.009242` (`5/1/8`) | `+0.019628` (`2/8/0`) | `-0.014854` (`1/7/2`) | `-0.00522` | `52.34 s / 24.95 s` |
| `vcd_blur2_a0.30` | `+0.015238` (`6/1/7`) | `-0.009074` (`8/1/5`) | `+0.017474` (`2/6/2`) | `-0.016998` (`1/5/4`) | `+0.00308` | `43.59 s / 24.96 s` |
| `vcd_blur4_a0.25` | `+0.006923` (`8/1/5`) | `+0.007101` (`8/1/5`) | `-0.003629` (`1/7/2`) | `-0.000615` (`2/6/2`) | `+0.00701` | `38.01 s / 24.95 s` |
| **`ilms_blank_a0.15`** | `+0.002345` (`4/1/9`) | **`+0.012836` (`8/1/5`)** | **`-0.001961` (`0/9/1`)** | **`+0.000740` (`1/8/1`)** | `+0.00759` | `34.29 s / 19.42 s` |
| **`ilms_blank_a0.30`** | `+0.000521` (`6/1/7`) | **`+0.026784` (`8/0/6`)** | `-0.003629` (`1/7/2`) | **`+0.002101` (`3/6/1`)** | **`+0.01365`** | **`27.91 s / 19.33 s`** |

## Scrutiny & Anomaly Check

1. **Coverage & Integrity:** All `48` pages × `6` arms scored cleanly (`0` decode failures, `0` NaN/Inf, `missing_keys=[]`).
2. **Strong, Monotonic Signal from Internal-LM Subtraction (`ilms_blank`) vs Noisy `vcd_blur`:**
   - **`ilms_blank` scales cleanly with `alpha` on photo domains:** `hw_photo` jumps **`+0.0128`** (`a0.15`, `8/1/5`) → **`+0.0268`** (`a0.30`, `8/0/6`, top wins `+0.2019` on `hw-debussy-vetement-du-blesse` and `+0.1197` on `hw-debussy-la-plus-que-lente`), while `pr_photo` also improves (`+0.0007` → `+0.0021`, `3/6/1`)!
   - **`ilms_blank_a0.15`** strictly clears the `PLAN.md` PROMOTE numbers on `hw_photo` (`+0.0128 >= +0.012`, `8 > 5`), `pr_clean` (`-0.0020 >= -0.003`), and `pr_photo` (`+0.0007`), while **`ilms_blank_a0.30`** doubles the `hw_photo` gain to **`+0.0268`** and misses `pr_clean >= -0.003` by only `0.0006` (`-0.0036`, driven by `-0.0196` on `schoenberg-op19-2` and `-0.0175` on `beethoven-grosse-fuge`).
   - **`vcd_blur` is unstable and discarded:** `vcd_blur2_a0.30` suffered a `-0.3689` collapse on `hw-beethoven-das-blumchen-wunderhold-d-b` photo, and `vcd_blur2_a0.15` regressed `pr_photo` by `-0.0149` (`-0.1445` on `mozart-k458-1`) and `hw_clean` by `-0.0197` (`3/1/10`) at ~2× latency.
3. **Deployment / Latency & Checkpoint Caveats:**
   - Running unconditional `B=2` step-by-step decoding without MTP-4 added `+50%` latency on printed (`19.3 s` vs `12.9 s`), even though `ilms_blank_a0.30` only changed `1.3` tokens/page on `pr_clean` (vs `42.1` tokens/page on `hw_photo`)!
   - Evaluated on `iter10_5_best` (`val_ser=0.3143`, T4 FP32) rather than `iter10_soup_baseline` (`val_ser=0.1923`).

## Verdict: **`ITERATE`**

- **Discard `vcd_blur`** (unstable, ~2× latency, printed photo regressions).
- **Iterate on `ilms_blank` (`a0.15..0.30`):** Strongest inference-time `hw_photo` win of the batch (`+0.0268` at `a0.30`, `+0.0128` at `a0.15` with zero printed-photo regression), but needs confidence-gating / pitch-duration token-gating + validation on `iter10_soup_baseline` / Iter 12 `v2` before production promotion so we capture the full `+0.0268` gain with `0.000` printed-clean dip and `< 5%` latency overhead.

## Follow-Up Action / Successor Experiment

- **Proposed Successor:** `i13-18b-conf-gated-ilms-blank` (depends on `2026-11-26T07-40-i13-18-trocr-pylaia-vcd-ilms`):
  1. Test **confidence-gated & token-gated `ilms_blank`** (`alpha in {0.15, 0.25, 0.30, 0.40}`, applying blank-LM subtraction only when `p_max < 0.85` and only on pitch/duration/accidental tokens, leaving structural tokens uncontrasted) on L4 BF16 against `checkpoints/iter10_soup_baseline/uni_omr_soup.pt`.
  2. Cache the single constant blank-page encoder memory once per session and measure composition with `i13-17`'s `vrfm_band_0.30`.
