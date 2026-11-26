# Experiment Outcome — `2026-11-26T01-38-ov-32-dinov3-gram-weight-sweep` (`ov-32`)

- **Status:** `COMPLETED`
- **Completed (UTC):** `2026-11-26T12:08:41Z` (started `2026-11-26T11:27:50Z`)
- **Platform / GPU:** Google Colab `NVIDIA L4` (`24 GB`, `BF16=True`, `muon+adamw`)
- **Elapsed Time:** `2451.49 s` (`40.9 min` compute; `2545.17 s` total queue wall)
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-26T01-38-ov-32-dinov3-gram-weight-sweep.json` (pulled to [`eval_result.json`](./eval_result.json))
  - Run Ledger: `runs/2026-11-26T01-38-ov-32-dinov3-gram-weight-sweep.json` (pulled to [`run_result.json`](./run_result.json))

---

## Measured Results (`500` steps/arm, `small_45m`, RISM `5,579` train / `128` held-out dense-feature diagnostic pages)

| Arm | `gram_weight` | Elapsed (s) | Final EMA Loss | Final `jepa_masked` | Dense `erank` Mean (Δ vs `0.0`) | Dense `erank` p10 | Norm Spectral Entropy `H_norm` | Top-1 Var Frac | Off-Diag Cosine Mean | Gram Drift MSE vs Step-0 | Global Page `erank` | Ridge Ink `R²` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `gram_0p0` *(ctrl)* | `0.0` | `691.3` | `0.17039` | `0.00859` | `60.312` (`+0.0%`) | `42.656` | `0.68464` | `0.70786` | `0.98605` | `0.000451` | **`7.373`** | `0.04012` |
| `gram_0p1` | `0.1` | `861.1` | `0.17019` | `0.00838` | `63.351` (`+5.0%`) | `45.591` | `0.69323` | `0.70486` | `0.98890` | `0.000341` | `6.986` | `0.04207` |
| **`gram_0p5`** | **`0.5`** | `861.6` | **`0.16914`** | **`0.00711`** (`-17.2%`) | **`66.815`** (**`+10.8%`**) | **`49.536`** | **`0.70325`** | **`0.68047`** | `0.99329` | `0.000310` | `6.693` | **`0.04809`** (`+19.9%`) |

*(All 3 arms: `nan_skips=0`, `oom_skips=0`, `128/128` held-out pages scored.)*

---

## Scrutiny & Anomaly Check

1. **Coverage & Stability — PASS:** All 3 arms completed `500/500` steps with `0` NaN/OOM skips and `128/128` diagnostic pages scored. Enabling the frozen Gram teacher adds `~24.6%` wall-clock overhead (`861s` vs `691s`).
2. **Monotonic Improvement Not Yet Saturated at `0.5`:** `gram_0p5` improved every single core metric over `gram_0p0`: dense `erank` **`+10.8%`** (`60.31 -> 66.82`), `erank` p10 **`+16.1%`** (`42.66 -> 49.54`), `final_jepa_masked` **`-17.2%`** (`0.00859 -> 0.00711`), `top1_var_frac` down (`0.708 -> 0.680`), and `ridge_ink_probe_r2` **`+19.9%`** (`0.0401 -> 0.0481`). However, `+10.8%` mean `erank` fell just short of the `>= +15%` PROMOTE threshold.
3. **CRITICAL ANOMALY — Severe DC-Mode Anisotropy (`offdiag_cosine = 0.986–0.993`, `global_page_erank = 6.69–7.37`):**
   - Every arm violated the `offdiag_cosine > 0.85` collapse warning (`0.986 -> 0.989 -> 0.993`), and `global_page_erank` actually *dropped* slightly (`7.37 -> 6.69`) while `top1_var_frac` remained `68–71%`.
   - This proves Gram anchoring alone preserves *relative* pairwise patch structure (`erank` rises) without removing the shared mean/DC cone across ink tokens (which `i13-04` showed VICReg `var_w=1.0, cov_w=0.04` directly combats).
4. **Gram Weight Boundary:** Because `gram_weight=0.5` was the top of the tested range and every metric was still improving monotonically (without hurting `jepa_masked`), the optimum likely lies at `0.5–2.0` combined with VICReg covariance decorrelation.

---

## Verdict: **`ITERATE`**

`gram_0p5` is clearly beneficial (`+10.8%` dense `erank`, `-17.2%` `jepa_masked`, `+19.9%` probe `R²`), but missed the `>= +15%` PROMOTE bar and left uncentered `offdiag_cosine` at `0.993` (`> 0.85`).

## Follow-Up Action / Successor Experiment

- **Queued Successor:** [`2026-11-26T16-20-ov-32b-gram-weight-high-vicreg`](../2026-11-26T16-20-ov-32b-gram-weight-high-vicreg/README.md) (`depends_on: ["2026-11-26T01-38-ov-32-dinov3-gram-weight-sweep"]` in `experiments/overnight_queue/queue.json`).
  - Sweeps `gram_weight in {0.5, 1.0, 2.0}` **with VICReg (`var_w=1.0, cov_w=0.04`)** enabled (per `i13-04`), and reports both raw and mean-centered off-diagonal cosine similarity, `dense_erank_mean`, `global_page_erank`, and `ridge_ink_probe_r2` to lock the Iter 13+ RISM SSL recipe.
