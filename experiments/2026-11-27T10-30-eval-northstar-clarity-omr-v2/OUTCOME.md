# OUTCOME — `2026-11-27T10-30-eval-northstar-clarity-omr-v2`

- **Status:** `COMPLETED` (`rc=0`)
- **Verdict:** **`RECORD / BASELINE`** — Audited `omr-northstar db12da9` benchmark baseline for **Clarity-OMR** (`clquwu/Clarity-OMR`, #2 external public contender behind `homr 0.7.1`, #3 overall behind `UniOMR Iter 14 Final` and `homr 0.7.1`). All 125 raw `.musicxml` outputs are archived on HF Private Hub for instant ~4 s CPU rescoring on future scorer updates.
- **Completed At:** `2026-11-27T12:49:11Z` (started `2026-11-27T11:50:50Z`)
- **Platform / GPU:** Google Colab `NVIDIA A100-SXM4-40GB` (session cleanly terminated; verified `0` lingering Clarity sessions)
- **Elapsed Time:** `4,146.9 s` total Colab job wall-clock (`3,500.63 s` eval script runtime; `3,403.1 s` cumulative per-page inference: **mean `27.22 s/page`**, median `25.46 s/page`, p95 `58.68 s/page`, min `5.58 s`, max `76.51 s`; `0` timeouts)
- **Scorer Version:** `gregoreesmaa/omr-northstar` commit **`db12da9`** (`validate_report` = **`0` errors**)
- **HF Private Hub Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Canonical 4-layer report: `evals/2026-11-27T10-30-eval-northstar-clarity-omr-v2.json` (local: [eval_hf.json](file:///C:/Users/grego/projects/sheet-music-converter/experiments/2026-11-27T10-30-eval-northstar-clarity-omr-v2/eval_hf.json))
  - Per-page breakdown: `evals/2026-11-27T10-30-eval-northstar-clarity-omr-v2_per_page.json` (local: [per_page_hf.json](file:///C:/Users/grego/projects/sheet-music-converter/experiments/2026-11-27T10-30-eval-northstar-clarity-omr-v2/per_page_hf.json))
  - Raw 125-page MusicXML archive: `evals/2026-11-27T10-30-eval-northstar-clarity-omr-v2_musicxml.tar.gz` (verified: **`125 / 125` non-empty `.musicxml` files**)
  - Run metadata ledger: `runs/2026-11-27T10-30-eval-northstar-clarity-omr-v2.json` (local: [run_meta.json](file:///C:/Users/grego/projects/sheet-music-converter/experiments/2026-11-27T10-30-eval-northstar-clarity-omr-v2/run_meta.json))

---

## 1. Apples-to-Apples Leaderboard Comparison (`omr-northstar db12da9`, 125 Pages)

| System | Scorer | **NS** `[95% CI]` | **`NS_tier`** | **CDR** (CFR) | **ALS** (CFR) | **RWP** (CFR) | **HWM** (CFR) | **`S_note`** | **`S_struct`** | **`S_expr`** | **`S_play`** | **`CFR_max`** | **`med_wFPP`** | **`NTSR`** | **`NMHR`** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Iter 14 Final (`UniOMR v2a`)** | `db12da9` | **32.37** `[23.80, 41.32]` | 11.97 | **85.64** (6.5%) | **85.13** (9.7%) | **85.24** (12.9%) | **22.91** (75.0%) | **0.7478** | **0.8350** | **0.5405** | **0.7264** | **75.0%** | **64.0** | **+0.612** | **0.0%** |
| **`homr 0.7.1`** *(AGPL-3.0)* | `db12da9` | **15.92** `[12.89, 19.83]` | **40.06** | 82.62 (6.5%) | 42.19 (51.6%) | 83.55 (3.2%) | 7.53 (96.4%) | 0.6161 | 0.6597 | 0.3613 | 0.4529 | 96.4% | 160.0 | +0.319 | 0.0% |
| **`Clarity-OMR v2`** *(this run)* | `db12da9` | **8.45** `[7.25, 9.46]` | 16.91 | 47.50 (32.3%) | 22.65 (74.2%) | 39.25 (32.3%) | 4.60 (100.0%) | 0.3506 | 0.5163 | 0.1238 | 0.1980 | 100.0% | 670.5 | -0.125 | 0.0% |
| `Clarity-OMR v1` *(pre-audit ref)* | `89c698c` | 8.73 `[7.38, 9.99]` | 19.71 | 49.26 (45.7%) | 19.50 (71.0%) | 38.19 (48.4%) | 5.70 (100.0%) | 0.3474 | 0.5652 | 0.1519 | 0.2342 | 100.0% | 550.0 | +0.113 | 0.0% |
| `Audiveris 5.11.0` *(AGPL-3.0)* | `db12da9` | 2.71 `[1.72, 3.50]` | 7.72 | 74.88 (19.4%) | 2.70 (100.0%) | 35.34 (71.0%) | 0.36 (100.0%) | 0.3413 | 0.3548 | 0.1728 | 0.2649 | 100.0% | 479.5 | +0.169 | 0.0% |
| `oemer 0.1.8` *(MIT)* | `db12da9` | 0.84 `[0.44, 1.14]` | 1.44 | 14.26 (96.8%) | 1.23 (100.0%) | 1.52 (100.0%) | 0.65 (100.0%) | 0.0575 | 0.0690 | 0.0422 | 0.0151 | 100.0% | 791.0 | -0.187 | 0.0% |

**Deltas vs key references (on `db12da9`):**
- **Iter 14 Final vs Clarity-OMR v2:** **`+23.92 NS`** (**3.83×**), `+38.14 CDR`, `+62.48 ALS`, `+45.99 RWP`, `+18.31 HWM`, `+0.3972 S_note`, `+0.3187 S_struct`, `+0.4167 S_expr`, `+0.5284 S_play`, **`-606.5 med_wFPP`** (`64.0` vs `670.5`), `+0.737 NTSR`. Iter 14 Final wins **`119 / 121` (98.3%)** music pages!
- **homr 0.7.1 vs Clarity-OMR v2:** `homr` leads by `+7.47 NS` (`15.92` vs `8.45`), `+23.15 NS_tier`, `+35.12 CDR`, `+19.54 ALS`, `+44.30 RWP`, `+2.93 HWM`, `-510.5 med_wFPP`.
- **Clarity-OMR v2 (`db12da9`) vs v1 (`89c698c`):** `-0.28 NS` (`8.45` vs `8.73`), `-2.80 NS_tier` (`16.91` vs `19.71`), `-0.238 NTSR` (`-0.125` vs `+0.113`, `med_wFPP` `670.5` vs `550.0`), while `CDR`/`RWP` CFR dropped from `45.7%`/`48.4%` to `32.3%`/`32.3%` and `ALS` Q rose `+3.15` (`22.65` vs `19.50`) under the audited `db12da9` CER scorer (which separates T0 negative controls out of CDR and tightened wFPP/NTSR accounting).

---

## 2. Per-Domain & Error Taxonomy Breakdown (`Clarity-OMR v2`)

| Domain | Music Pages | Stratum Q | Stratum CFR | Non-CF Pages | `S_note` | `S_struct` | `S_expr` | `S_play` | Mean `n_pred` / `n_gt` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **CDR** | 31 | **47.50** | 32.3% | 21 / 31 | 0.5915 | 0.6565 | 0.2200 | 0.3515 | 204.9 / 213.2 |
| **ALS** | 31 | **22.65** | 74.2% | 8 / 31 | 0.2576 | 0.5171 | 0.1165 | 0.1512 | 147.3 / 213.2 |
| **RWP** | 31 | **39.25** | 32.3% | 21 / 31 | 0.5113 | 0.5944 | 0.1437 | 0.2676 | 204.2 / 213.2 |
| **HWM** | 28 | **4.60** | 100.0% | 0 / 28 | 0.0090 | 0.2736 | 0.0035 | 0.0028 | 120.2 / 404.4 |

- **Negative Controls (`4 / 4` PASS, `0.0%` NMHR):** `neg-blank-staff-01/02` and `neg-program-notes-01/02` all returned valid 0-note MusicXML in `5.63–5.67 s/page`.
- **Layer 3 Error Taxonomy:** Layout/segmentation `27.0%`, pitch/accidental `24.5%`, engraving/expressive `19.4%`, hallucination/truncation `13.6%`, global context drift `8.7%`, rhythm/meter `6.8%`, polyphony routing `0.03%`.
- **Catastrophic Failure Reasons (121 music pages, non-exclusive):** `dropped_system`: `66` (`9` CDR, `21` ALS, `8` RWP, `28` HWM); `structural_collapse`: `46` (`2` CDR, `17` ALS, `3` RWP, `24` HWM); `empty_output`: `12` (`1` ALS, `11` HWM); `hallucination_loop`: `12` (`1` ALS, `11` HWM). `50 / 121` music pages avoided catastrophic failure.

---

## 3. Competitor-Differential Analysis: Where Clarity-OMR v2 Beats Iter 14 Final

Across all 121 music pages, **Clarity-OMR v2 beats Iter 14 Final on `Q` on only `2 / 121` pages** — both are the same T2 lead-sheet work:

| Rank | Page ID | Domain / Tier | `ΔQ` (Clarity − Iter 14) | Clarity-OMR v2 (`Q` / `S_note` / `S_struct` / `wFPP` / `n` / `m`) | Iter 14 Final (`Q` / `S_note` / `S_struct` / `wFPP` / `n` / `m` / CF) |
|---:|---|---|---:|---|---|
| **1** | `cdr-foster-leadsheet` | `CDR / T2` | **`+73.23`** | `Q=76.59`, `S_note=0.984`, `S_struct=0.934`, `wFPP=26.0`, `n=94/95`, `m=35/35`, `CF=[]` | `Q=3.36`, `S_note=0.021`, `S_struct=0.460`, `wFPP=395.0`, `n=1/95`, `m=2/35`, `CF=[dropped_system, structural_collapse]` |
| **2** | `rwp-foster-leadsheet` | `RWP / T2` | **`+67.17`** | `Q=70.53`, `S_note=0.989`, `S_struct=0.684`, `wFPP=26.5`, `n=93/95`, `m=35/35`, `CF=[]` | `Q=3.36`, `S_note=0.021`, `S_struct=0.460`, `wFPP=395.0`, `n=1/95`, `m=2/35`, `CF=[dropped_system, structural_collapse]` |
| *3 (closest)* | `hwm-clean-hw-hensel-im-herbste-d-b` | `HWM / T3` | `-0.60` *(Iter 14 wins Q; Clarity `+0.043 S_note`)* | `Q=7.97`, `S_note=0.050`, `S_struct=0.279`, `wFPP=2012.0`, `n=209/354`, `m=183/54` | `Q=8.57`, `S_note=0.007`, `S_struct=0.392`, `wFPP=1041.0`, `n=202/354`, `m=39/54` |

*(Additional `S_note`-only spot where Clarity edges Iter 14: `cdr-essen-ballad` (`CDR/T1`), `S_note=0.959` vs `0.944` (`+0.014`), though Iter 14 still wins `Q` decisively at `91.95` vs `73.25`.)*

### Key Concrete Engineering Lesson for Iter 15
- **Single-work T2 collapse explains Iter 14 Final's entire `NS_tier` deficit (`11.97` vs Clarity `16.91` and homr `40.06`):** Because `foster-leadsheet` is the sole T2 work in OMR-NorthStar v0.1, Iter 14 Final's premature EOS (`1/95` notes, `2/35` measures) on lead-sheet layouts with dense chord symbols + lyrics above single-staff melodies drags `NS_tier` below both `homr` and `Clarity-OMR`.
- **Why Clarity-OMR aced `foster-leadsheet` (`0.984–0.989 S_note`):** Stage-A YOLO staff cropping isolates each staff strip independently, so chord-symbol/title/lead-sheet text spacing never triggers premature full-page EOS.
- **Actionable takeaway for Iter 15:** Fixing lead-sheet / chord-symbol / single-staff-with-lyrics early EOS (via band-tiling / system-strip verification / T2 lead-sheet synthetic coverage + `i15-03` lyrics/text separation + `i15-44` verifier KV-rewind) is an immediate, high-leverage `NS_tier` unlock (+25–30 `NS_tier` points on a single layout class).

---

## 4. Scrutiny & Anomaly Check

1. **Coverage & validity:** `125 / 125` pages scored (`121` music + `4` NC), `125 / 125` (`100.0%`) parseable MusicXML, `125 / 125` non-empty `.musicxml` archived in HF `_musicxml.tar.gz`, `0` conversion failures, `0` timeouts, `0` NaNs, and **`validate_report` = `0` errors** on `omr-northstar db12da9`.
2. **Apples-to-apples sanity:** Exact same 125-page pack (`datasets/northstar_eval_pack_125p.tar.gz`) and exact same `omr-northstar db12da9` scorer as `Iter 14 Final`, `homr 0.7.1`, `Audiveris 5.11.0`, and `oemer 0.1.8`.
3. **Domain asymmetry:** Clarity-OMR is reasonably strong on clean/phone monophonic & lead-sheet scores (`CDR 47.50`, `RWP 39.25`, up to `0.989 S_note`), degrades sharply on archival scans (`ALS 22.65`, `74.2%` CFR), and collapses completely on handwritten manuscripts (`HWM 4.60`, `S_note 0.0090`, `100.0%` CFR).
4. **Hardware / platform quirks:** Ran smoothly on Colab A100-40GB (`27.22 s/page` mean, nearly identical to L4's `27.54 s/page` because Clarity-OMR is CPU/subprocess-bound on per-page Python startup + PyMuPDF + beam-5 decoding); session cleanly stopped.

---

## 5. Follow-Up Action

- **Status:** Baseline recorded and archived on HF (`evals/2026-11-27T10-30-eval-northstar-clarity-omr-v2_musicxml.tar.gz`). No retraining or code promotion required for this external GPL-3.0 baseline.
- **Competitor-differential takeaway fed into Iter 15:** Address T2 lead-sheet early-EOS collapse (`cdr/rwp/als-foster-leadsheet`) via band-tiling/strip verification (`i15-05`, `i15-44`) and lyrics/chord-symbol decoupling (`i15-03`).
- **Pending batch commit:** Per coordinator instructions, `queue.json` and git commit/push are deferred to the upcoming batch commit alongside the new P0 experiments `i15-41..50`.
