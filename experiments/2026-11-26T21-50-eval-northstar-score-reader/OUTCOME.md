# OUTCOME — `2026-11-26T21-50-eval-northstar-score-reader`

- **Status:** `COMPLETED` *(Competitor Baseline Recorded)*
- **Verdict:** **`DISCARD`** *(Pure competitor benchmark — recorded in OMR-NorthStar leaderboard; not a UniOMR training change)*
- **Target System:** `score-reader==0.1.8` (`Luckolite/Sheet-Music-Recognition`, **Apache-2.0** code & weights — TensorFlow/ONNX vision + Java 17 structural decoder → MusicXML 4.0)
- **Started / Completed:** `2026-11-27T15:36:17Z` → `2026-11-27T18:07:55Z`
- **Elapsed Time:** `9,097.57s` (`151.6 min` for 125 pages)
- **Platform / GPU:** `NVIDIA A100-SXM4-40GB` (Modal)
- **HF Artifacts:**
  - Canonical 4-Layer Report: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-26T21-50-eval-northstar-score-reader.json`
  - Per-Page Breakdown: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-26T21-50-eval-northstar-score-reader_per_page.json`
  - Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-26T21-50-eval-northstar-score-reader.json`

---

## Measured Results (OMR-NorthStar 125-Page Permissive Pack, `dev-v1`)

### Layer 1 — Headline & Stratum Scorecard
| Metric | `score-reader==0.1.8` | Notes |
|---|---:|---|
| **Headline `NS`** | **`6.22`** | 95% CI `[4.46, 7.81]` (dominated by worst stratum `HWM` & `CFR_max`) |
| `tier_weighted_ns` | `15.79` | — |
| `m_geom` | `14.94` | — |
| `worst_stratum_q` | `3.37` | `HWM` |
| `cfr_max` | `96.77%` | `ALS` (`30/31` catastrophic failures) |
| **CDR** (Clean Digital, `n=35`) | **`61.63`** | `31/35` parseable, `CFR = 25.71%` (`9/35`) |
| **RWP** (Real Phone Photo, `n=31`) | **`34.15`** | `26/31` parseable, `CFR = 58.06%` (`18/31`), `Δ_photo = -32.15` |
| **ALS** (Archival Scan, `n=31`) | **`7.02`** | `18/31` parseable, `CFR = 96.77%` (`30/31`) |
| **HWM** (Handwritten MS, `n=28`) | **`3.37`** | `9/28` parseable, `CFR = 82.14%` (`23/28`) |

### Macro Axes, Human Utility, Negative Controls & Efficiency
| Category | Metric | Value |
|---|---|---:|
| **Axes (Macro)** | `S_note` / `S_struct` / `S_expr` / `S_play` | `0.3294` / **`0.4463`** / `0.1296` / `0.2904` |
| **Human Utility** | `median_wfpp` / `p90_wfpp` / `p99_wfpp` | `588.0` / `1,697.2` / `3,805.9` |
| **Human Utility** | `mean_ntsr` / `nupr` | `-1.1773` / `0.632` |
| **Pillar 3B Neg. Controls** | `NMHR` / Pass Rate (`n=4`) | **`0.00%`** (`4/4` PASSED — `0` notes on blank staves & program notes) |
| **Conversion** | `conversion_ok` / `conversion_fail` | `84 / 125` (`67.2%`) / `41 / 125` (`32.8%`) |
| **Efficiency** | `median_latency_s` / `p95_latency_s` | `69.78s` / `108.11s` |

### Tier × Domain Highlights (`CDR` vs `RWP` vs `ALS` vs `HWM`)
- **Strong on clean printed engraving (`CDR`):** `T1 = 73.57` (`n=8`), `T2 = 77.65` (`n=1`), `T3 = 70.74` (`n=11`), `T4 = 65.95` (`n=6`), `T5 = 43.02` (`n=5`). Top pages reached `91.43` (`cdr-fiddle-strathspey`, `wfpp=22.0`), `91.17` (`cdr-mozart-k458-1`, `wfpp=32.5`), `87.44` (`cdr-bach-bwv66.6`, `wfpp=49.5`), and even `84.33` on `rwp-bach-bwv66.6`.
- **Severe collapse on archival & handwritten:** `ALS` collapsed across every tier (`T3 = 0.72`, `T1 = 12.10`, `T4 = 10.13`, `T5 = 9.86`), and `HWM` collapsed to `3.91` (`T3`) and `2.66` (`T4`).

---

## Scrutiny & Anomaly Check

1. **Full 125/125 Coverage & Faithful Failure Accounting:** All 125 pages ran to completion (`84` conversion OK, `41` CLI/empty-XML failures logged faithfully as `parseable=False`, `Q=0.0`, zero cherry-picking).
2. **Classic Hybrid CV/Neural Fragility Profile:** Score-Reader's staff-line detector + Java structural decoder excels on clean high-contrast digital engravings (`CDR = 61.63`, `S_struct = 0.4463`) and cleanly abstains on non-music/blank pages (`NMHR = 0.00%`, `4/4` pass), but degrades sharply under phone-photo warp/lighting (`Δ_photo = -32.15`) and breaks almost completely on archival scans (`ALS CFR = 96.77%`) and historical handwriting (`HWM CFR = 82.14%`).
3. **Latency Bottleneck:** `69.78s` median / `108.11s` p95 per page (total `2.53 hr` for 125 pages) due to per-page JVM + TensorFlow invocation overhead — roughly `15–30×` slower than `UniOMR v2`.
4. **Harness Telemetry Gaps (Non-Blocking):** `layer3_error_taxonomy` reports all `0.0` (known stdlib scorer placeholder), and `peak_memory_gb = 0.0` / `parameter_count = 0` because subprocess memory and model params were not instrumented.

---

## Verdict Rationale & Follow-Up Action

- **Verdict (`DISCARD` as training candidate; `RECORDED` as OMR-NorthStar competitor baseline):** Score-Reader (`NS = 6.22`) is now officially logged on the OMR-NorthStar leaderboard.
- **Follow-Up (`learn-from-northstar`):**
  1. Log `score-reader==0.1.8` (`NS = 6.22`, `CDR = 61.63`, `RWP = 34.15`, `ALS = 7.02`, `HWM = 3.37`, `NMHR = 0.00%`, `69.8s/page`) into `progress.md` and the OMR-NorthStar leaderboard.
  2. Because Score-Reader is **Apache-2.0**, inspect its structural MusicXML decoder (`S_struct = 0.4463`, `CDR = 61.63`, `NMHR = 0.00%`) during the `learn-from-northstar` forensic pass for any clean bar-completion or blank-page abstention heuristics applicable to `UniOMR v2` / `mscz-rs`.
