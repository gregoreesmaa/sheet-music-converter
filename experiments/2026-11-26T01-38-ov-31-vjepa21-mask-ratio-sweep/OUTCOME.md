# Experiment Outcome — `2026-11-26T01-38-ov-31-vjepa21-mask-ratio-sweep` (`ov-31`)

- **Status:** `COMPLETED`
- **Completed (UTC):** `2026-11-26T12:13:23Z` (started `2026-11-26T11:29:58Z`)
- **Platform / GPU:** Google Colab `NVIDIA L4` (`24 GB`, `BF16=True`, `muon+adamw`)
- **Elapsed Time:** `2604.96 s` (`43.4 min` compute; `2920.2 s` total queue wall)
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-26T01-38-ov-31-vjepa21-mask-ratio-sweep.json` (pulled to [`eval_result.json`](./eval_result.json))
  - Run Ledger: `runs/2026-11-26T01-38-ov-31-vjepa21-mask-ratio-sweep.json` (pulled to [`run_result.json`](./run_result.json))

---

## Measured Results (`500` steps/arm, `small_45m`, RISM `5,451` train / `256` val probe: `192` fit / `64` test)

| Arm | `mask_ratio` (range) | Realized `masked_ink_frac` | Final EMA Loss | Tail-100 Loss Std | Ridge Staff `R²` (Δ vs `0.45`) | kNN-5 Staff `R²` (Δ vs `0.45`) | kNN-5 Quartile Acc | Ridge Ink `R²` | NaN / OOM Skips |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mask_045` *(ctrl)* | `0.45` (`[0.40, 0.50]`) | `0.459` | **`0.16620`** | **`0.00575`** | **`0.21028`** (`+0.00000`) | `0.15631` (`+0.00000`) | `0.26562` | **`0.11871`** | `0 / 0` |
| `mask_060` | `0.60` (`[0.55, 0.65]`) | `0.614` | `0.17034` | `0.00654` | `0.19696` (`-0.01332`) | `0.16215` (`+0.00584`) | **`0.34375`** | `0.11199` | `0 / 0` |
| `mask_075` | `0.75` (`[0.70, 0.80]`) | `0.804` | `0.17565` | `0.00933` | `0.17897` (`-0.03131`) | **`0.23187`** (**`+0.07556`**) | `0.32812` | `0.10668` | `0 / 0` |

---

## Scrutiny & Anomaly Check

1. **Coverage & Stability — PASS:** All 3 arms completed `500/500` steps cleanly (`855–857 s` each) with `0` NaN skips and `0` OOM skips. All `256` held-out RISM validation pages (`192` fit / `64` test) were scored with zero NaNs.
2. **Sharp Divergence Between Linear Ridge and Cosine kNN-5 Probes:**
   - **Linear Ridge `R²`** monotonically *declines* as masking increases (`0.21028 -> 0.19696 -> 0.17897`, `-0.03131` at `0.75`), and `ridge_ink_r2` follows the same downward slope (`0.11871 -> 0.11199 -> 0.10668`).
   - **Cosine `k=5` kNN `R²`** surges at `mask_075` (`0.15631 -> 0.16215 -> 0.23187`, **`+0.07556`** vs baseline), while kNN quartile accuracy peaks at `mask_060` (`0.34375`, `+7.81 pp`).
3. **Realized Mask Overshoot & Variance at `0.75`:** Because `mask_075` actually masks `80.4%` of ink tokens (`final_masked_ink_frac=0.804`), tail-100 loss std jumps `1.62×` (`0.00933` vs `0.00575`). Meanwhile, `ov-31` ran without VICReg anti-collapse (`var_w=1.0, cov_w=0.04` proven essential in `i13-04`), so global linear geometry (`Ridge R²`) may be harmed by anisotropy at aggressive constant masking from step 1.
4. **Hardware / Platform:** Zero anomalies on Colab L4 BF16.

---

## Verdict: **`ITERATE`**

Matches `PLAN.md` Section 6 (`ITERATE`: `0.75` strongly improves kNN-5 probe `R²` by `+0.0756` but hurts linear Ridge `R²` by `-0.0313` and increases tail variance `1.62×`). Neither constant `0.45` nor constant `0.75` dominates both linear and neighborhood structure.

## Follow-Up Action / Successor Experiment

- **Queued Successor:** [`2026-11-26T16-20-ov-31b-vjepa-mask-curriculum-vicreg`](../2026-11-26T16-20-ov-31b-vjepa-mask-curriculum-vicreg/README.md) (`depends_on: ["2026-11-26T01-38-ov-31-vjepa21-mask-ratio-sweep"]` in `experiments/overnight_queue/queue.json`).
  - Tests intermediate constant `mask_065` (`[0.60, 0.70]`) and a linear **mask curriculum `0.45 -> 0.70`** against `mask_045` control (500 steps/arm on RISM 5,707), with **VICReg anti-collapse (`var_w=1.0, cov_w=0.04`)** enabled so linear Ridge `R²`, kNN-5 `R²`, and dense `erank` all improve together.
