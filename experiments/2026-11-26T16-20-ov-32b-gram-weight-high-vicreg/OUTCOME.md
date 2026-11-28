# Experiment Outcome — `2026-11-26T16-20-ov-32b-gram-weight-high-vicreg` (`ov-32b`)

- **Status:** `COMPLETED`
- **Parent Experiment:** [`2026-11-26T01-38-ov-32-dinov3-gram-weight-sweep`](../2026-11-26T01-38-ov-32-dinov3-gram-weight-sweep/OUTCOME.md) (`ITERATE`)
- **Completed (UTC):** `2026-11-28T02:51:24Z` (started `2026-11-28T02:35:28Z`)
- **Platform / GPU:** Google Colab `NVIDIA A100-SXM4-40GB` (`BF16=True`, `muon+adamw`)
- **Elapsed Time:** `955.29 s` (`15.9 min` compute across 3 arms; `1401.44 s` total queue wall)
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-26T16-20-ov-32b-gram-weight-high-vicreg.json` (pulled to [`evals/2026-11-26T16-20-ov-32b-gram-weight-high-vicreg.json`](./evals/2026-11-26T16-20-ov-32b-gram-weight-high-vicreg.json))
  - Run Ledger: `runs/2026-11-26T16-20-ov-32b-gram-weight-high-vicreg.json` (pulled to [`runs/2026-11-26T16-20-ov-32b-gram-weight-high-vicreg.json`](./runs/2026-11-26T16-20-ov-32b-gram-weight-high-vicreg.json))

---

## Measured Results (`500` steps/arm, `small_45m`, VICReg `var_w=1.0, cov_w=0.04`, RISM `5,579` train / `128` held-out dense-feature diagnostic pages)

| Arm | `gram_weight` | VICReg (`var/cov`) | Elapsed (s) | Peak VRAM (GB) | Final EMA Loss (`Best`) | Final `jepa_masked` | Dense `erank` Mean (Δ vs `0.5`) | Dense `erank` p10 | `H_norm` | Top-1 Var Frac | Off-Diag Cosine Mean (`<=0.85` gate) | Gram Drift MSE vs Step-0 | Global Page `erank` (`>=15.0` gate) | Ridge Ink `R²` |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| *Parent `ov-32` `gram_0p5`* | `0.5` | `0.0 / 0.0` *(off)* | `861.6` *(L4)* | — | `0.16914` | `0.00711` | `66.815` | `49.536` | `0.70325` | `0.68047` | `0.99329` *(FAIL)* | `0.000310` | `6.693` *(FAIL)* | `0.04809` |
| **`gram_0p5`** | **`0.5`** | **`1.0 / 0.04`** | `309.60` | `13.67` | `0.82151` (`0.80758`) | `0.20176` | **`182.025`** (`+0.0%`) | **`156.012`** | **`0.87381`** | **`0.11387`** | **`0.78753`** (**PASS**) | `0.040337` | **`55.114`** (**PASS**) | `0.12223` |
| `gram_1p0` | `1.0` | `1.0 / 0.04` | `305.20` | `13.67` | **`0.80784`** (**`0.79604`**) | `0.14807` | `166.910` (`-8.3%`) | `141.008` | `0.85914` | `0.13344` | `0.85506` *(FAIL)* | `0.017572` | `50.092` *(PASS)* | **`0.12791`** |
| `gram_2p0` | `2.0` | `1.0 / 0.04` | `305.42` | `13.67` | `0.80894` (`0.79667`) | **`0.10958`** | `154.432` (`-15.2%`) | `126.697` | `0.84589` | `0.15955` | `0.90187` *(FAIL)* | **`0.007404`** | `46.109` *(PASS)* | `0.12147` |

*(All 3 arms: `optimizer_mode="muon+adamw"`, `nan_skips=0`, `oom_skips=0`, `128/128` held-out RISM diagnostic pages scored, `1.68–1.71 steps/s`.)*

---

## Scrutiny & Anomaly Check

1. **Coverage & Stability — PASS:** All 3 arms ran `500/500` steps with `0` NaN skips, `0` OOM skips, `13.67 GB` peak VRAM, and `128/128` diagnostic pages scored. No silent truncations, missing keys, or optimizer fallbacks (`muon+adamw` verified on all 3 arms).
2. **VICReg Eliminates the `ov-32` DC-Mode Collapse (`+172.4%` Dense `erank`, `8.23×` Global Page `erank`):**
   - Comparing `gram_0p5` with VICReg (`1.0 / 0.04`) against parent `ov-32` `gram_0p5` without VICReg:
     - `dense_erank_mean`: `66.815 → 182.025` (**`+172.4%`**, `2.72×`)
     - `dense_erank_p10`: `49.536 → 156.012` (**`+214.9%`**, `3.15×`)
     - `dense_top1_var_frac`: `0.68047 → 0.11387` (**`-56.66 pp`** — dominant DC cone eliminated)
     - `dense_offdiag_cosine_mean`: `0.99329 → 0.78753` (**`-0.20576`**, clearing the `<= 0.85` gate)
     - `global_page_erank`: `6.693 → 55.114` (**`8.23×`**, clearing the `>= 15.0` gate by `3.67×`)
     - `ridge_ink_probe_r2`: `0.04809 → 0.12223` (**`2.54×`**)
3. **Why Higher `gram_weight` (`1.0`, `2.0`) Strictly Hurts Geometry Under VICReg:**
   - Because `gram_teacher` is a **frozen step-0 random-init encoder**, cranking `gram_weight` from `0.5 → 1.0 → 2.0` pulls the learned student representations back toward the step-0 random Gram matrix (`gram_drift_vs_step0_mse`: `0.040337 → 0.017572 → 0.007404`).
   - This monotonically degrades every representational diversity metric: `dense_erank_mean` falls `182.025 → 166.910 → 154.432` (`-15.2%`), `dense_erank_p10` falls `156.012 → 141.008 → 126.697`, `top1_var_frac` rises `0.11387 → 0.13344 → 0.15955`, `global_page_erank` falls `55.114 → 50.092 → 46.109`, and `dense_offdiag_cosine_mean` breaches the `<= 0.85` gate on both `gram_1p0` (`0.85506`) and `gram_2p0` (`0.90187`).
   - **Collapse-proxy trap verified:** `final_jepa_masked` is lower at `gram_2p0` (`0.10958` vs `0.20176`) precisely because tokens become more collinear (`offdiag_cosine = 0.902`), making masked prediction artificially easier without improving semantic readout (`ridge_ink_probe_r2` is flat/worse at `0.12147` vs `0.12223`).
4. **Minor Telemetry & Platform Notes:**
   - `run_colab.py` did not emit `vicreg_loss` as a separate key in `history` (though `vicreg_var_weight=1.0, vicreg_cov_weight=0.04` were active in `vjepa21_forward_loss`, confirmed by `loss ~ 0.81` vs `ov-32`'s `0.17` and the `2.72×` erank jump).
   - Ran on Colab `A100-SXM4-40GB` (`955.29 s` compute) instead of `L4` (`~40 min` est), `2.57×` faster than `ov-32` on L4 (`2451.49 s`).

---

## Verdict: **`PROMOTE`** (`gram_weight=0.5` + VICReg `var_w=1.0, cov_w=0.04`; **`DISCARD`** `gram_weight` `1.0` & `2.0`)

`gram_0p5` is the **only arm** to pass all three `PLAN.md` promotion gates — and clears them by huge margins:
- `dense_erank_mean = 182.025 >= 70.0` (**`2.60×` gate**)
- `dense_offdiag_cosine_mean = 0.78753 <= 0.85` (**PASS**; `1.0` fails at `0.85506`, `2.0` fails at `0.90187`)
- `global_page_erank = 55.114 >= 15.0` (**`3.67×` gate**)

## Follow-Up Action / Successor Experiment

- **Immediate Implementation (`PROMOTE`):** Updated default `--gram-weight` from `1.0` to `0.5` in [`tmp-train/mvd/pretrain_ijepa_rism_v2.py`](../../tmp-train/mvd/pretrain_ijepa_rism_v2.py) (locking the `PLAN.md` promotion while preserving `--disable-random-gram` for scratch runs and pairing with `--vicreg-var-weight 1.0 --vicreg-cov-weight 0.04` for RISM SSL pretraining).
- **No further `gram_weight > 0.5` sweep needed:** `1.0` and `2.0` are definitively discarded.
