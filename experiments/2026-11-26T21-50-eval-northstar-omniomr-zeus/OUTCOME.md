# OUTCOME — `2026-11-26T21-50-eval-northstar-omniomr-zeus`

- **Status:** `COMPLETED` (`exit_code=0`, Competitor Baseline Recorded)
- **Verdict:** **`DISCARD`** *(Pure competitor baseline — recorded on the OMR-NorthStar leaderboard; no upstream UniOMR model integration)*
- **Target System:** **OmniOMR Zeus** (`OmniOMR/zeus`, **MIT** code; snapshot `zeus-camera-grandstaff-lmx-1.0-2024-02-12.model`, **CC-BY-SA-4.0**, TensorFlow 2.12 in isolated Python 3.10 `uv` venv + OpenCV horizontal staff-projection system cropper → Linearized MusicXML `LMX` → MusicXML)
- **Started / Completed:** `2026-11-27T20:36:10Z` → `2026-11-27T21:04:39Z` (Colab job wall-clock: `2026-11-27T20:08:58Z` → `2026-11-27T21:04:45Z`)
- **Elapsed Time:** `1,709.83s` (`28.5 min`) eval script runtime; `3,347.34s` (`55.8 min`) total Colab wall-clock including Py3.10 venv + TF 2.12 setup
- **Platform / GPU:** Google Colab `NVIDIA A100-SXM4-40GB` (session cleanly terminated; verified `0` active OmniOMR sessions)
- **Scorer Version:** Bundled `omr_northstar` stdlib scorer (`1.0.0-local-pack`, `dev-v1`)
- **HF Private Hub Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Canonical 4-Layer Report: `evals/2026-11-26T21-50-eval-northstar-omniomr-zeus.json` (local: [eval_hf.json](file:///C:/Users/grego/projects/sheet-music-converter/experiments/2026-11-26T21-50-eval-northstar-omniomr-zeus/eval_hf.json))
  - Per-Page Breakdown: `evals/2026-11-26T21-50-eval-northstar-omniomr-zeus_per_page.json` (local: [eval_hf_per_page.json](file:///C:/Users/grego/projects/sheet-music-converter/experiments/2026-11-26T21-50-eval-northstar-omniomr-zeus/eval_hf_per_page.json))
  - Run Ledger: `runs/2026-11-26T21-50-eval-northstar-omniomr-zeus.json` (local: [run_hf.json](file:///C:/Users/grego/projects/sheet-music-converter/experiments/2026-11-26T21-50-eval-northstar-omniomr-zeus/run_hf.json))

---

## 1. Apples-to-Apples Leaderboard Comparison (OMR-NorthStar 125-Page Pack, `dev-v1`)

| System | Scorer | **NS** `[95% CI]` | **`NS_tier`** | **CDR** (CFR) | **ALS** (CFR) | **RWP** (CFR) | **HWM** (CFR) | **`S_note`** | **`S_struct`** | **`S_expr`** | **`S_play`** | **`CFR_max`** | **`med_wFPP`** | **`NTSR`** | **`NMHR`** | **Med Lat** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Iter 14 Final (`UniOMR v2a`)** | `db12da9` | **32.37** `[23.80, 41.32]` | 11.97 | **85.64** (6.5%) | **85.13** (9.7%) | **85.24** (12.9%) | **22.91** (75.0%) | **0.7478** | **0.8350** | **0.5405** | **0.7264** | **75.0%** | **64.0** | **+0.612** | **0.0%** | — |
| **`homr 0.7.1`** *(AGPL-3.0)* | `db12da9` | **15.92** `[12.89, 19.83]` | **40.06** | 82.62 (6.5%) | 42.19 (51.6%) | 83.55 (3.2%) | 7.53 (96.4%) | 0.6161 | 0.6597 | 0.3613 | 0.4529 | 96.4% | 160.0 | +0.319 | 0.0% | — |
| `Clarity-OMR v2` *(GPL-3.0)* | `db12da9` | 8.45 `[7.25, 9.46]` | 16.91 | 47.50 (32.3%) | 22.65 (74.2%) | 39.25 (32.3%) | 4.60 (100.0%) | 0.3506 | 0.5163 | 0.1238 | 0.1980 | 100.0% | 670.5 | -0.125 | 0.0% | 25.46s |
| `score-reader 0.1.8` *(Apache-2.0)* | `local-pack` | 6.22 `[4.46, 7.81]` | 15.79 | 61.63 (25.7%) | 7.02 (96.8%) | 34.15 (58.1%) | 3.37 (82.1%) | 0.3294 | 0.4463 | 0.1296 | 0.2904 | 96.8% | 588.0 | -1.177 | 0.0% | 69.78s |
| `Audiveris 5.11.0` *(AGPL-3.0)* | `db12da9` | 2.71 `[1.72, 3.50]` | 7.72 | 74.88 (19.4%) | 2.70 (100.0%) | 35.34 (71.0%) | 0.36 (100.0%) | 0.3413 | 0.3548 | 0.1728 | 0.2649 | 100.0% | 479.5 | +0.169 | 0.0% | — |
| **`OmniOMR Zeus`** *(this run, MIT / CC-BY-SA-4.0)* | `local-pack` | **1.50** `[0.95, 1.92]` | **2.50** | **10.13**\* (94.3%) | **1.54** (96.8%) | **2.74** (96.8%) | **3.45** (100.0%) | **0.0340** | **0.1761** | **0.0501** | **0.0204** | **100.0%** | **750.0** | **-0.266** | **25.0%** | **15.08s** |
| `oemer 0.1.8` *(MIT)* | `db12da9` | 0.84 `[0.44, 1.14]` | 1.44 | 14.26 (96.8%) | 1.23 (100.0%) | 1.52 (100.0%) | 0.65 (100.0%) | 0.0575 | 0.0690 | 0.0422 | 0.0151 | 100.0% | 791.0 | -0.187 | 0.0% | — |

*\*`local-pack` scorer folds the 4 T0 negative controls into `CDR` (`n=35`); on the 31 CDR music pages alone, OmniOMR Zeus scores `Q = 4.99` (`CFR = 100.0%`).*

**Deltas vs key references:**
- **Iter 14 Final (`UniOMR v2a`) vs OmniOMR Zeus:** **`+30.87 NS`** (**21.6×**), `+9.47 NS_tier`, `+75.51 CDR`, `+83.59 ALS`, `+82.50 RWP`, `+19.46 HWM`, `+0.7138 S_note`, `+0.6589 S_struct`, `+0.4904 S_expr`, `+0.7060 S_play`, **`-686.0 med_wFPP`** (`64.0` vs `750.0`), `+0.878 NTSR`, `-25.0 pp NMHR`.
- **`homr 0.7.1` vs OmniOMR Zeus:** **`+14.42 NS`** (**10.6×**), `+37.56 NS_tier`, `+72.49 CDR`, `+40.65 ALS`, `+80.81 RWP`, `+4.08 HWM`, `-590.0 med_wFPP`, `-25.0 pp NMHR`.

---

## 2. Per-Domain, Negative Control & Efficiency Breakdown

| Domain | Pages (`n`) | Parseable (`n_ok`) | Nonzero-Note Music Pages | Stratum `Q` (Music-Only) | Stratum `CFR` | `S_note` | `S_struct` | `S_expr` | `S_play` | Mean `n_pred` / `n_gt` | Mean `m_pred` / `m_gt` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **CDR** | 35 (31 mus + 4 NC) | 27 / 35 | 22 / 31 | **10.13** (`4.99`) | 94.3% (`100.0%`) | 0.1140 | 0.2295 | 0.0933 | 0.0699 | 77.6 / 213.2 | 9.3 / 47.4 |
| **ALS** | 31 | 28 / 31 | 12 / 31 | **1.54** | 96.8% | 0.0007 | 0.1817 | 0.0323 | 0.0010 | 12.5 / 213.2 | 6.6 / 47.4 |
| **RWP** | 31 | 25 / 31 | 24 / 31 | **2.74** | 96.8% | 0.0053 | 0.1378 | 0.0645 | 0.0011 | 75.8 / 213.2 | 8.2 / 47.4 |
| **HWM** | 28 | 26 / 28 | 24 / 28 | **3.45** | 100.0% | 0.0025 | 0.1455 | 0.0000 | 0.0014 | 48.2 / 404.4 | 7.3 / 90.1 |
| **Total** | **125** | **106 / 125 (84.8%)** | **82 / 121 (67.8%)** | `M_geom = 3.48` | `CFR_max = 100.0%` | **0.0340** | **0.1761** | **0.0501** | **0.0204** | — | — |

- **Conversion & Coverage:** `106 / 125` (`84.8%`) pages produced parseable MusicXML; `19 / 125` (`15.2%`) failed LMX/MusicXML conversion (`parseable=False`, logged faithfully as `Q = 0.0`, `unparseable` CF).
- **Catastrophic Failure Reasons (125 pages, non-exclusive):** `structural_collapse`: `119`, `dropped_system`: `43`, `empty_output`: `39`, `unparseable`: `19`, `hallucination_loop`: `6`. Only `2 / 121` music pages avoided catastrophic failure (`rwp-ryans-hornpipe` `Q=10.20`, `als-repeats-melody` `Q=9.86`).
- **Pillar 3B Negative Controls (`NMHR = 25.0%`, `3 / 4` PASS — Gate FAILED):**
  - `neg-program-notes-01` / `neg-program-notes-02`: **PASS** (`0` notes, `Q=100.0`, `0.06s` — OpenCV cropper detected 0 staff bands and fast-pathed empty MusicXML).
  - `neg-blank-staff-01`: **PASS** on note hallucination (`0` notes), but `parseable=False` (`Q=0.0`, `15.74s` — LMX decode failure on blank staves).
  - `neg-blank-staff-02`: **FAIL** (`pred_note_count=1` hallucinated note on blank staff, `Q=0.0`, `14.50s`).
- **Top Music Pages (Confirming Native Piano Grand-Staff Specialization):**
  1. `cdr-joplin-maple-leaf` (`CDR / T3`, piano grand-staff): **`Q = 51.81`**, **`S_note = 0.820`**, `S_struct = 0.607`, `S_expr = 0.167`, `S_play = 0.198`, `n = 445 / 518`, `m = 50 / 58`, `wFPP = 475.5`.
  2. `rwp-essen-ballad` (`RWP / T1`): `Q = 27.70`, `S_note = 0.000`, `S_struct = 0.308`, `S_expr = 1.000`, `n = 39 / 36`, `wFPP = 124.5`.
  3. `cdr-haydn-op74n1-1` (`CDR / T4`): `Q = 18.58`, `S_note = 0.175`, `S_struct = 0.464`, `n = 65 / 198`, `wFPP = 579.5`.
  4. `cdr-chopin-mazurka06-2` (`CDR / T3`, piano grand-staff): `Q = 15.49`, `S_note = 0.234`, `S_struct = 0.246`, `n = 112 / 359`, `wFPP = 956.0`.
  5. `cdr-clara-polonaise-op1n1` (`CDR / T3`, piano grand-staff): `Q = 12.91`, `S_note = 0.267`, `S_struct = 0.379`, `n = 295 / 508`, `wFPP = 1238.5`.
- **Efficiency:** `median_latency_s = 15.08s`, `mean = 13.33s`, `p95 = 16.12s`, `min = 0.06s`, `max = 67.99s` (TensorFlow 2.12 subprocess startup dominates per-page latency at ~15s/page).

---

## 3. Scrutiny & Anomaly Check

1. **Full 125/125 Coverage & Faithful Failure Logging:** All 125 pages executed to completion (`106` parseable MusicXML, `19` conversion/LMX failures faithfully scored as `parseable=False`, `Q=0.0`; `0` NaNs, `0` timeouts).
2. **Domain / Architecture Mismatch (Primary Root Cause of `NS = 1.50`):** `zeus-camera-grandstaff-lmx-1.0` is a **system-level piano grand-staff recognizer**, not an end-to-end full-page multi-instrumentation OMR model. Its `S_note = 0.820` (`445/518` notes, `50/58` measures) on `cdr-joplin-maple-leaf` proves the model and LMX→MusicXML pipeline work cleanly when the OpenCV horizontal staff-projection cropper isolates clean grand-staff systems, but it collapses (`m_pred = 6.6–9.3` vs `47.4–90.1` GT, `119/125` `structural_collapse`) on single-staff melodies, SATB chorales, string quartets, orchestral pages, archival scans (`ALS = 1.54`), phone photos (`RWP = 2.74`), and historical manuscripts (`HWM = 3.45`).
3. **Scorer Version Asymmetry (`local-pack` vs `db12da9`):** `run_colab.py` used the bundled `1.0.0-local-pack` stdlib scorer (pre-`db12da9`), which folds the 4 T0 negative controls into `CDR` (inflating `CDR` from `4.99` music-only to `10.13` via the two `Q=100.0` program-notes pages) and leaves `layer3_error_taxonomy` at `0.0`. Headline `NS = 1.50` is still anchored by `worst_stratum_q = ALS = 1.54` and `CFR_max = 100.0%`, so the leaderboard rank (6th of 7, between `Audiveris 5.11.0` `2.71` and `oemer 0.1.8` `0.84`) is unaffected.
4. **Platform / Hardware Notes:** Originally queued for Modal L4 (`modal run --detach` failed early with `rc=1`), then cleanly executed on Colab `A100-SXM4-40GB` in `55.8 min` total (`28.5 min` eval runtime) using an isolated Python 3.10 `uv` venv for TensorFlow 2.12. `peak_memory_gb = 0.0` and `parameter_count = 0` are uninstrumented subprocess placeholders.

---

## 4. Verdict Rationale & Follow-Up Action

- **Verdict (`DISCARD` as training candidate; `RECORDED` as OMR-NorthStar competitor baseline):** OmniOMR Zeus (`NS = 1.50`) is officially logged on the OMR-NorthStar leaderboard.
- **Key Engineering Takeaway:** System-level grand-staff LMX models paired with heuristic horizontal-projection croppers cannot generalize to full-page multi-instrumentation OMR (`ALS/RWP/HWM` collapse, `15.2%` LMX parse failures, `25.0%` NMHR on blank staves), validating `UniOMR v2`'s end-to-end full-page architecture (`NS = 32.37`, **21.6×** higher).
- **Follow-Up:** No successor training run required. Queue entry `2026-11-26T21-50-eval-northstar-omniomr-zeus` marked `COMPLETED` in `experiments/overnight_queue/queue.json`.
