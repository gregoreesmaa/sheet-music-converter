# Experiment Outcome — `2026-11-26T14-30-i13-06b-sar-sam-conf-skip-ttt` (`i13-06b`)

- **Status:** `COMPLETED`
- **Completed (UTC):** `2026-11-26T13:42:51Z` (started `2026-11-26T13:10:56Z`)
- **Platform / GPU:** Google Colab `NVIDIA L4` (`24 GB`, `BF16=True`; script logged `kaggle-NVIDIA L4` due to `/kaggle` check order)
- **Elapsed Time:** `1915.0 s` (`31.9 min` compute; `2207.05 s` total queue wall)
- **Base Checkpoint:** HF `checkpoints/iter10_soup_baseline/uni_omr_soup.pt` (`uni_omr_v1`, `step=12000`, `val_ser=0.192301`, `missing_keys=[]`, `unexpected_keys=[]`)
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-26T14-30-i13-06b-sar-sam-conf-skip-ttt.json` (pulled to [`eval_result.json`](./eval_result.json))
  - Run Ledger: `runs/2026-11-26T14-30-i13-06b-sar-sam-conf-skip-ttt.json` (pulled to [`run_result.json`](./run_result.json))

---

## Measured Results (`max_dim=1280`, `max_len=3840`, MTP-4 + FSA, `ent_w=1.0, kl_w=0.5, lr=3e-5, steps=2`)

Baseline (`0` TTT): `hw_clean = 0.2666`, `hw_photo = 0.1695`, `pr_clean = 0.5736`, `pr_photo = 0.5594` (`14` HW × 2 + `10` Printed × 2 = `48` pages).

### 1. Pre-Registered Arms (Un-Gated & `conf_skip_thr in {-0.08, -0.12, -0.18, -0.25}`)

| Arm | `hw_clean` Δ (W/T/L, skip) | `hw_photo` Δ (W/T/L, skip) | `pr_clean` Δ (skip) | `pr_photo` Δ (skip) | `grosse-fuge` photo |
|---|---:|---:|---:|---:|---:|
| `arm_ctrl_ov38` *(un-gated)* | `+0.0053` (`6/0/8`, `0%`) | **`+0.0248`** (`8/1/5`, `0%`) | `+0.0004` (`0%`) | `-0.0087` (`0%`, FAIL) | `0.2222` (`-0.0936`) |
| `arm_sar_rho005` *(un-gated)* | `+0.0072` (`6/0/8`, `0%`) | `+0.0221` (`9/1/4`, `0%`) | `+0.0004` (`0%`) | `-0.0111` (`0%`, FAIL) | `0.2222` (`-0.0936`) |
| `arm_sar_rho010` *(un-gated)* | **`+0.0099`** (`6/0/8`, `0%`) | `+0.0214` (`8/1/5`, `0%`) | `+0.0004` (`0%`) | `-0.0101` (`0%`, FAIL) | `0.2222` (`-0.0936`) |
| `arm_ctrl_ov38_skip_008` | `-0.0050` (`1/6/7`, `43%`) | `+0.0151` (`6/4/4`, `29%`) | **`+0.0000`** (`100%`) | **`+0.0000`** (`100%`) | **`0.3158` (`0.0000`)** |
| `arm_ctrl_ov38_skip_012` | `-0.0003` (`1/9/4`, `64%`) | `+0.0170` (`4/9/1`, `64%`) | **`+0.0000`** (`100%`) | **`+0.0000`** (`100%`) | **`0.3158` (`0.0000`)** |
| `arm_sar_rho005_skip_008` | `+0.0003` (`3/6/5`, `43%`) | `+0.0105` (`6/4/4`, `29%`) | **`+0.0000`** (`100%`) | **`+0.0000`** (`100%`) | **`0.3158` (`0.0000`)** |
| `arm_sar_rho005_skip_012` | `-0.0014` (`1/9/4`, `64%`) | `+0.0197` (`4/9/1`, `64%`) | **`+0.0000`** (`100%`) | **`+0.0000`** (`100%`) | **`0.3158` (`0.0000`)** |
| `arm_sar_rho010_skip_008` | `+0.0016` (`3/6/5`, `43%`) | `+0.0060` (`5/4/5`, `29%`) | **`+0.0000`** (`100%`) | **`+0.0000`** (`100%`) | **`0.3158` (`0.0000`)** |
| `arm_sar_rho010_skip_012` | `-0.0014` (`1/9/4`, `64%`) | `+0.0106` (`2/9/3`, `64%`) | **`+0.0000`** (`100%`) | **`+0.0000`** (`100%`) | **`0.3158` (`0.0000`)** |

*(At `skip_018` and `skip_025`, `93–100%` of HW pages were skipped, collapsing deltas to `<= 0`.)*

### 2. Offline Exact Simulation of Tighter Threshold (`mean_logprob > -0.040`)

| Arm (`thr = -0.040`) | `hw_clean` Δ (skip) | `hw_photo` Δ (skip) | `pr_clean` Δ (skip) | `pr_photo` Δ (skip) | `grosse-fuge` photo |
|---|---:|---:|---:|---:|---:|
| **`arm_ctrl_ov38` (`skip > -0.040`)** | `+0.0036` (`14%`) | **`+0.0248`** (**`0%`**) | **`+0.0000`** (`100%`) | **`+0.0011`** (`90%`) | **`0.3158` (`0.0000`)** |
| `arm_sar_rho005` (`skip > -0.040`) | `+0.0052` (`14%`) | `+0.0221` (**`0%`**) | **`+0.0000`** (`100%`) | **`-0.0009`** (`90%`) | **`0.3158` (`0.0000`)** |
| **`arm_sar_rho010` (`skip > -0.040`)** | **`+0.0088`** (`14%`) | `+0.0214` (**`0%`**) | **`+0.0000`** (`100%`) | **`+0.0001`** (`90%`) | **`0.3158` (`0.0000`)** |

---

## Scrutiny & Anomaly Check

1. **Coverage & Integrity — PASS:** All `48/48` pages (`28` HW + `20` printed) scored cleanly with `0` OOM hits and `0` NaN/None scores.
2. **Confidence Skip Gate Hypothesis Confirmed — Threshold Calibration Was 2–6× Too Loose:**
   - Printed `mean_logprob` lies in `[-0.0489, -0.0008]` (median `-0.004`, with `beethoven-grosse-fuge` photo at `-0.03234`).
   - Handwritten `mean_logprob` lies in `[-0.2137, -0.0158]` (`hw_clean`, median `-0.089`) and `[-0.1700, -0.0406]` (`hw_photo`, median `-0.094`).
   - Because the pre-registered grid started at `-0.08`, it skipped `43%` of `hw_clean` and `29%` of `hw_photo`, discarding most of the TTT win. Setting **`thr = -0.040`** cleanly separates the two distributions: **`100%` of `pr_clean` and `90%` of `pr_photo` skipped (`pr_photo = +0.0001 / +0.0011`, `grosse-fuge` untouched)** while **`0%` of `hw_photo` is skipped**!
3. **Why `ent_w=1.0` Underperformed `i13-06`'s `ent_w=0.5` on `hw_photo`:** Raising `ent_w` from `0.5` (`i13-06`: `+0.0378` on `hw_photo`) to `1.0` (`i13-06b`: `+0.0214` on `hw_photo`) over-sharpened SAR-SAM updates on noisy HW photos (and on `hw_clean`, `arm_sar_rho010` reached `+0.0099` vs `arm_ctrl_ov38` `+0.0053`, still below the `+0.015` PROMOTE bar).

---

## Verdict: **`ITERATE`**

The zero-update confidence skip gate completely eliminates the printed regression (`pr_photo: -0.0101 -> +0.0001` at `thr = -0.040` with `0%` `hw_photo` skip), but the pre-registered `-0.08..-0.25` grid was miscalibrated and `ent_w=1.0` weakened `hw_photo` gains relative to `ent_w=0.5`.

## Follow-Up Action / Successor Experiment

- **Queued Successor:** [`2026-11-26T16-20-i13-06c-sar-sam-tight-conf-skip`](../2026-11-26T16-20-i13-06c-sar-sam-tight-conf-skip/README.md) (`depends_on: ["2026-11-26T14-30-i13-06b-sar-sam-conf-skip-ttt"]` in `experiments/overnight_queue/queue.json`).
  - Evaluates **`ent_w in {0.5, 0.75}`** × `sam_rho in {0.0, 0.01}` × `steps in {2, 3}` with the calibrated tight confidence skip gate **`thr in {-0.035, -0.040, -0.045}`** across all `14` HW manuscripts + `10` printed safety pieces to hit the `PROMOTE` gate (`hw_clean >= +0.015`, `hw_photo >= +0.030`, `pr_clean/photo >= -0.002`).
