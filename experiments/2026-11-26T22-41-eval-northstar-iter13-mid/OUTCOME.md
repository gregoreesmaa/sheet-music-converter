# OUTCOME — `2026-11-26T22-41-eval-northstar-iter13-mid`

- **Status:** `COMPLETED` (`PROMOTE / NEW BEST v2 MILESTONE`)
- **Completed (UTC):** `2026-11-27T03:02:13Z` (started `2026-11-27T02:50:17Z`)
- **Platform / GPU:** Google Colab `NVIDIA L4` (`24 GB`, `BF16=True`, peak VRAM `0.768 GB`)
- **Elapsed Time:** `715.95 s` (`11m 56s` eval compute; `1010.14 s` queue wall)
- **Target System:** `uni-omr-iter13-v2-step744-ema` (`uni_omr_v2`, `d_model=384`, `n_heads=8`, `n_kv_heads=2`, `max_mem_tokens=3072`, `vocab_size=237`, `41,334,795` params, step `744` EMA)
- **Checkpoint:** `checkpoints/iter13_v2_mid/uni_omr_v2_ema_latest.pt`
- **Export Backend:** `rust:hf:assembly-serialize-linux-x86_64` (`121/121` music pages via static Linux Rust `mscz-rs`, `4/4` negative controls via `neg-gate`; `needs_local_rust_rescore = false`)
- **Inference Config:** `max_dim=1408`, `max_len=3584`, `bf16=True`, `use_grammar=True`, `use_mtp_spec=True`, `neg_control_gate=True`
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Summary JSON: `evals/2026-11-26T22-41-eval-northstar-iter13-mid.json` (pulled to [`2026-11-26T22-41-eval-northstar-iter13-mid.json`](./2026-11-26T22-41-eval-northstar-iter13-mid.json))
  - Per-Page JSON: `evals/2026-11-26T22-41-eval-northstar-iter13-mid_per_page.json` (pulled to [`2026-11-26T22-41-eval-northstar-iter13-mid_per_page.json`](./2026-11-26T22-41-eval-northstar-iter13-mid_per_page.json))
  - Bekern JSON: `evals/2026-11-26T22-41-eval-northstar-iter13-mid_bekern.json` (pulled to [`2026-11-26T22-41-eval-northstar-iter13-mid_bekern.json`](./2026-11-26T22-41-eval-northstar-iter13-mid_bekern.json))
  - Run Ledger: `runs/2026-11-26T22-41-eval-northstar-iter13-mid.json` (pulled to [`run_ledger_2026-11-26T22-41-eval-northstar-iter13-mid.json`](./run_ledger_2026-11-26T22-41-eval-northstar-iter13-mid.json))

---

## Side-by-Side Comparison (100% Apples-to-Apples Rust `mscz-rs` Exporter, OMR-NorthStar v0.1 125p)

| Metric | Iter 10.5 (`v1`, 8k EMA, `mscz-rs`) | Iter 12 (`v2`) | **Iter 13 Mid (`v2`, step 744 EMA)** | Delta (`Iter13-mid - Iter12`) | Delta (`Iter13-mid - Iter10.5`) |
|---|---:|---:|---:|---:|---:|
| **NS Score (95% CI)** | 32.36 `[26.93, 37.86]` | 26.75 | **31.88** `[25.59, 39.07]` *(31.8840)* | **+5.13** | -0.47 |
| **Tier-Weighted NS (`Tier-W`)** | 22.95 | — | **41.66** *(41.6552)* | — | **+18.71** *(1.82×)* |
| **Geometric Mean (`M_geom`)** | 52.05 | — | **55.37** *(55.3693)* | — | **+3.32** |
| **Worst Stratum Q (`HWM`)** | 20.63 | 18.77 | **21.84** *(21.8399)* | **+3.07** | **+1.21** |
| **`CFR_max` (HWM)** | 53.57% | — | **64.29%** *(0.6429)* | — | +10.71 pp |
| **CDR (Clean Digital)** | 77.44 | 80.16 | **79.69** *(79.6885)* | -0.47 | **+2.25** |
| **ALS (Archival / Low-Res Scan)** | 61.95 | 51.13 | **70.08** *(70.0798)* | **+18.95** | **+8.13** |
| **RWP (Real-World Photo)** | 74.13 | 76.28 | **77.06** *(77.0618)* | **+0.78** | **+2.93** |
| **HWM (Handwritten Manuscript)** | 20.63 | 18.77 | **21.84** *(21.8399)* | **+3.07** | **+1.21** |
| **`S_note` (macro)** | 0.6576 | 0.6265 | **0.7016** | **+0.0751** | **+0.0440** |
| **`S_struct` (macro)** | 0.7417 | — | **0.7812** | — | **+0.0395** |
| **`S_expr` (macro)** | 0.4403 | 0.4434 | **0.4599** | **+0.0165** | **+0.0196** |
| **`S_play` (macro)** | 0.6516 | — | **0.7014** | — | **+0.0498** |
| **`mean_NTSR`** | +0.4631 | +0.403 | **+0.5114** | **+0.108** | **+0.0483** |
| **Median `wFPP` / p90 / p99** | `224.0` / `1276.5` / `3062.8` | — | **`174.5`** / `1261.3` / `2705.6` | — | **-49.5** *(better)* |
| **`NUPR`** | 21.6% | — | **16.8%** *(0.168)* | — | **-4.8 pp** *(better)* |
| **Negative Control (`NMHR`)** | **0.0%** (`4/4` PASS) | **0.0%** (`4/4` PASS) | **0.0%** (`4/4` PASS) | `0.0 pp` | `0.0 pp` |
| **Paired `Δ_photo`** | -0.3982 | — | **-0.0059** | — | **+0.3923** |
| **Median / p95 Latency** | `3.78 s` / `9.93 s` | — | `5.03 s` / `12.18 s` | — | `+1.25 s` |

---

## Per-Domain Breakdown (`uni-omr-iter13-v2-step744-ema`)

| Domain | Pages (Music / NC) | Stratum Q | Music-Only Q | Stratum CFR | Music `S_note` | Music `S_struct` | Music `S_expr` | Music `S_play` | Music Median `wFPP` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **CDR** | 35 (31 / 4) | **79.69** | 77.07 | 17.1% | 0.8650 | 0.8669 | 0.5770 | 0.8863 | 103.5 |
| **ALS** | 31 (31 / 0) | **70.08** | 70.08 | 22.6% | 0.8132 | 0.8223 | 0.4904 | 0.7755 | 149.0 |
| **RWP** | 31 (31 / 0) | **77.06** | 77.06 | 22.6% | 0.8710 | 0.8557 | 0.5777 | 0.8937 | 112.5 |
| **HWM** | 28 (28 / 0) | **21.84** | 21.84 | 64.3% | 0.1668 | 0.5269 | 0.0887 | 0.1593 | 1213.5 |

### Per-Tier `mean_Q` (Music Pages)
- **T0 (4 NC pages in CDR):** `100.00` (`4/4` PASS, `0` hallucinated notes)
- **T1 (8 works × 3 domains):** `CDR 86.64`, `ALS 73.30`, `RWP 84.26`
- **T2 (1 work × 3 domains):** `CDR 52.50`, `ALS 78.02`, `RWP 61.47`
- **T3 (11 printed × 3 + 16 HWM):** `CDR 72.23`, `ALS 64.97`, `RWP 73.65`, `HWM 26.24`
- **T4 (6 printed × 3 + 12 HWM):** `CDR 95.47`, `ALS 87.11`, `RWP 94.27`, `HWM 15.98`
- **T5 (5 works × 3 domains):** `CDR 55.22`, `ALS 54.15`, `RWP 55.54`

---

## Scrutiny & Anomaly Check

1. **Coverage & Conversion Integrity — PASS:** All `125 / 125` pages scored cleanly (`conversion_ok = 125`, `conversion_fail = 0`, `parseable = 125/125`, `0` NaN/Inf across all per-page metrics). Negative-control gate passed `4 / 4` (`NMHR = 0.0%`).
2. **Apples-to-Apples Rust Exporter Parity — PASS:** Used static Linux Rust `assembly-serialize-linux-x86_64` (`mscz-rs`) on all `121/121` music pages (`needs_local_rust_rescore = false`), matching `Iter 12 v2` and `Iter 10.5 v1 (mscz-rs)` identically. Both `pred_raw` and stitched `bekern` are persisted on HF.
3. **Domain / Metric Asymmetry (Why `NS 31.88` trails `v1 32.36` by `0.47` despite winning every single stratum & axis):**
   - `Iter 13 mid` beats `Iter 10.5 v1 (mscz-rs)` on **every stratum** (`CDR +2.25`, `ALS +8.13`, `RWP +2.93`, `HWM +1.21`), **every axis** (`S_note +0.044`, `S_struct +0.040`, `S_expr +0.020`, `S_play +0.050`), `M_geom` (`55.37` vs `52.05`), `Tier-W NS` (`41.66` vs `22.95`, **+18.71** because `v1` collapses on `T2` to `2–4 Q`), `median_wFPP` (`174.5` vs `224.0`), `mean_NTSR` (`+0.511` vs `+0.463`), and `NUPR` (`16.8%` vs `21.6%`).
   - Raw `NS` is slightly capped by `cfr_max = 64.29%` on `HWM` (`18/28` HWM pages catastrophic failure vs `15/28` on `v1`; CF reasons across all 125p: `structural_collapse=23`, `hallucination_loop=13`, `dropped_system=12`).
   - Photo invariance is near-zero degradation (`Δ_photo = -0.0059`, `RWP S_note 0.8710 > CDR 0.8650`), and `ALS` jumped `+18.95` over `Iter 12 v2` (`51.13 -> 70.08`).
4. **Hardware / Platform:** Initial Modal detach attempt failed (`rc=1`); re-dispatched cleanly on Colab L4 (`BF16`, `0.768 GB` peak VRAM, session already terminated, zero idle CU burn).

---

## Verdict: **`PROMOTE` (NEW BEST `v2` MILESTONE)**

Even at **step 744** (early mid-Stage 2), `Iter 13 v2` decisively beats `Iter 12 v2` (`+5.13 NS`, `+18.95 ALS`, `+3.07 HWM`, `+0.075 S_note`, `+0.108 mean_NTSR`) and sets a new all-time high on `Tier-W NS` (`41.66`), `M_geom` (`55.37`), `ALS` (`70.08`), `RWP` (`77.06`), `HWM` (`21.84`), and all four macro axes (`S_note/S_struct/S_expr/S_play`).

## Follow-Up Action / Successor Experiment

- Feed per-page HWM failure forensics (`2026-11-26T22-41-eval-northstar-iter13-mid_per_page.json` + `_bekern.json`) into the `learn-from-northstar` forensic loop and Iter 14 training plan (specifically targeting `HWM CFR` `64.3% -> <30%` and `T5` `55 Q -> 75+ Q`).
- Re-evaluate final `Iter 13` Stage 2 EMA checkpoint on OMR-NorthStar v0.1 (`125p`) once local Stage 2 training finishes.
