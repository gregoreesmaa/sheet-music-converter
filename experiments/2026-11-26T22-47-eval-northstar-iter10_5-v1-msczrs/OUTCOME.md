# OUTCOME — `2026-11-26T22-47-eval-northstar-iter10_5-v1-msczrs`

- **Status:** `COMPLETED` (`RECORD / APPLES-TO-APPLES v1 BASELINE`)
- **Completed (UTC):** `2026-11-27T03:14:49Z` (started `2026-11-27T03:05:14Z`)
- **Platform / GPU:** Google Colab `NVIDIA L4` (`24 GB`, `BF16=True`, peak VRAM `0.920 GB`)
- **Elapsed Time:** `574.54 s` (`9m 35s` eval compute; `752.78 s` queue wall)
- **Target System:** `uni-omr-iter10_5-v1-8k-ema-msczrs` (`uni_omr_v1`, `backbone_stride=4x8`, `d_model=384`, `vocab_size=237`, `44,772,293` params, step `8,000` EMA)
- **Checkpoint:** `checkpoints/iter10_5_best/uni_omr_ema_best.pt`
- **Export Backend:** `rust:hf:assembly-serialize-linux-x86_64` (`121/121` music pages via static Linux Rust `mscz-rs`, `4/4` negative controls via `neg-gate`; `needs_local_rust_rescore = false`)
- **Inference Config:** `max_dim=1408`, `max_len=3584`, `bf16=True`, `use_grammar=True`, `use_mtp_spec=True`, `neg_control_gate=True`
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Summary JSON: `evals/2026-11-26T22-47-eval-northstar-iter10_5-v1-msczrs.json` (pulled to [`2026-11-26T22-47-eval-northstar-iter10_5-v1-msczrs.json`](./2026-11-26T22-47-eval-northstar-iter10_5-v1-msczrs.json))
  - Per-Page JSON: `evals/2026-11-26T22-47-eval-northstar-iter10_5-v1-msczrs_per_page.json` (pulled to [`2026-11-26T22-47-eval-northstar-iter10_5-v1-msczrs_per_page.json`](./2026-11-26T22-47-eval-northstar-iter10_5-v1-msczrs_per_page.json))
  - Bekern JSON: `evals/2026-11-26T22-47-eval-northstar-iter10_5-v1-msczrs_bekern.json` (pulled to [`2026-11-26T22-47-eval-northstar-iter10_5-v1-msczrs_bekern.json`](./2026-11-26T22-47-eval-northstar-iter10_5-v1-msczrs_bekern.json))
  - Run Ledger: `runs/2026-11-26T22-47-eval-northstar-iter10_5-v1-msczrs.json` (pulled to [`run_ledger_2026-11-26T22-47-eval-northstar-iter10_5-v1-msczrs.json`](./run_ledger_2026-11-26T22-47-eval-northstar-iter10_5-v1-msczrs.json))

---

## Exporter Parity Audit: Python Fallback (`21-56`) vs Rust `mscz-rs` (`22-47`) vs `Iter 13 Mid v2`

| Metric | Iter 10.5 `v1` (Python fallback `21-56`) | **Iter 10.5 `v1` (Rust `mscz-rs`, THIS RUN)** | Delta (`mscz-rs - py-fallback`) | Iter 13 Mid (`v2`, step 744, `mscz-rs`) |
|---|---:|---:|---:|---:|
| **NS Score (95% CI)** | 10.61 `[9.03, 12.13]` | **32.36** `[26.93, 37.86]` *(32.3552)* | **+21.75** *(3.05×)* | 31.88 `[25.59, 39.07]` |
| **Tier-Weighted NS (`Tier-W`)** | 12.21 | **22.95** *(22.9457)* | **+10.73** | **41.66** |
| **Geometric Mean (`M_geom`)** | 24.63 | **52.05** *(52.0464)* | **+27.42** | **55.37** |
| **Worst Stratum Q (`HWM`)** | 10.99 | **20.63** *(20.6309)* | **+9.64** | **21.84** |
| **`CFR_max` (HWM)** | 100.0% | **53.57%** *(0.5357)* | **-46.43 pp** | 64.29% |
| **CDR (Clean Digital)** | 39.04 | **77.44** *(77.4406)* | **+38.40** | **79.69** |
| **ALS (Archival / Low-Res Scan)** | 27.46 | **61.95** *(61.9548)* | **+34.49** | **70.08** |
| **RWP (Real-World Photo)** | 31.22 | **74.13** *(74.1314)* | **+42.91** | **77.06** |
| **HWM (Handwritten Manuscript)** | 10.99 | **20.63** *(20.6309)* | **+9.64** | **21.84** |
| **`S_note` (macro)** | 0.4265 | **0.6576** | **+0.2311** | **0.7016** |
| **`S_struct` (macro)** | 0.4473 | **0.7417** | **+0.2944** | **0.7812** |
| **`S_expr` (macro)** | 0.1299 | **0.4403** | **+0.3104** *(3.39×)* | **0.4599** |
| **`S_play` (macro)** | 0.5912 | **0.6516** | **+0.0604** | **0.7014** |
| **`mean_NTSR`** | +0.1105 | **+0.4631** | **+0.3526** | **+0.5114** |
| **Median `wFPP` / p90 / p99** | `725.5` / `1332.5` / `2546.7` | **`224.0`** / `1276.5` / `3062.8` | **-501.5** | **`174.5`** / `1261.3` / `2705.6` |
| **`NUPR`** | 60.0% | **21.6%** *(0.216)* | **-38.4 pp** | **16.8%** |
| **Negative Control (`NMHR`)** | **0.0%** (`4/4` PASS) | **0.0%** (`4/4` PASS) | `0.0 pp` | **0.0%** (`4/4` PASS) |
| **Paired `Δ_photo`** | +0.0397 | **-0.3982** | -0.4379 | **-0.0059** |
| **Median / p95 Latency** | `3.70 s` / `9.94 s` | `3.78 s` / `9.93 s` | `+0.08 s` | `5.03 s` / `12.18 s` |

---

## Per-Domain Breakdown (`uni-omr-iter10_5-v1-8k-ema-msczrs`)

| Domain | Pages (Music / NC) | Stratum Q | Music-Only Q | Stratum CFR | Music `S_note` | Music `S_struct` | Music `S_expr` | Music `S_play` | Music Median `wFPP` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **CDR** | 35 (31 / 4) | **77.44** | 74.53 | 11.4% | 0.8356 | 0.8219 | 0.5788 | 0.8480 | 97.5 |
| **ALS** | 31 (31 / 0) | **61.95** | 61.95 | 25.8% | 0.6917 | 0.7785 | 0.4560 | 0.6461 | 225.0 |
| **RWP** | 31 (31 / 0) | **74.13** | 74.13 | 16.1% | 0.8346 | 0.8276 | 0.5509 | 0.8487 | 102.0 |
| **HWM** | 28 (28 / 0) | **20.63** | 20.63 | 53.6% | 0.1777 | 0.4803 | 0.0672 | 0.1720 | 1257.2 |

### Per-Tier `mean_Q` (Music Pages)
- **T0 (4 NC pages in CDR):** `100.00` (`4/4` PASS, `0` hallucinated notes)
- **T1 (8 works × 3 domains):** `CDR 76.18`, `ALS 65.18`, `RWP 80.14`
- **T2 (1 work × 3 domains):** `CDR 4.13`, `ALS 2.09`, `RWP 4.18` *(complete collapse on `foster-leadsheet`)*
- **T3 (11 printed × 3 + 16 HWM):** `CDR 74.64`, `ALS 60.81`, `RWP 73.32`, `HWM 24.44`
- **T4 (6 printed × 3 + 12 HWM):** `CDR 95.16`, `ALS 74.40`, `RWP 91.36`, `HWM 15.55`
- **T5 (5 works × 3 domains):** `CDR 60.97`, `ALS 56.34`, `RWP 59.62`

---

## Scrutiny & Anomaly Check

1. **Coverage & Conversion Integrity — PASS:** All `125 / 125` pages scored cleanly (`conversion_ok = 125`, `conversion_fail = 0`, `parseable = 125/125`, `0` NaN/Inf). Negative-control gate passed `4 / 4` (`NMHR = 0.0%`).
2. **Critical Exporter Finding (Why `21-56` was invalidated):** On the exact same checkpoint and predictions, replacing the single-staff Python fallback exporter with the static Linux Rust `assembly-serialize-linux-x86_64` (`mscz-rs`) binary raised `NS` by **+21.75 points** (`10.61 -> 32.36`, **3.05×**), slashed `NUPR` from `60.0% -> 21.6%`, and cut `structural_collapse` from `100 -> 15` pages (`dropped_system=15`, `hallucination_loop=10`). **Rule confirmed:** never compare a Python-fallback NorthStar run against a Rust `mscz-rs` run; always download `tools/assembly-serialize-linux-x86_64` from HF.
3. **Domain / Tier Asymmetry:** `v1` (`Iter 10.5`) is surprisingly competitive on printed `T3`/`T4`/`T5` once multi-staff Rust export is active (`CDR 77.44`, `RWP 74.13`), and achieves slightly lower HWM CFR (`53.6%`, `15/28` pages) than `Iter 13 mid` (`64.3%`, `18/28`), which yields `NS 32.36` vs `31.88`. However, `v1` collapses completely on `T2` (`2.09–4.18 Q`, dragging `Tier-W NS` down to `22.95` vs `Iter 13 mid`'s `41.66`), lags heavily on `ALS` (`61.95` vs `70.08`), and trails `Iter 13 mid` on every single macro axis (`S_note`, `S_struct`, `S_expr`, `S_play`), `M_geom`, `median_wFPP`, and `mean_NTSR`.
4. **Hardware / Platform:** Initial Modal detach attempt failed (`rc=1`); re-dispatched cleanly on Colab L4 (`BF16`, `0.92 GB` peak VRAM, session already terminated, zero idle CU burn).

---

## Verdict: **`RECORD / BASELINE`** (Supersedes `2026-11-26T21-56-eval-northstar-iter10_5-v1`)

Locks in the true, 100% apples-to-apples Rust `mscz-rs` internal baseline for UniOMR `v1` (`Iter 10.5`, 8k EMA): **`NS = 32.36`**, **`Tier-W NS = 22.95`**, **`M_geom = 52.05`**, **`NMHR = 0.0%`**.

## Follow-Up Action / Successor Experiment

- Use this run (never `21-56`) as the official `v1` anchor row in all internal NorthStar progression tables alongside `Iter 12 v2`, `Iter 13 Mid v2` (`2026-11-26T22-41-eval-northstar-iter13-mid`), and upcoming `Iter 13 Final` / `Iter 14` evaluations.
