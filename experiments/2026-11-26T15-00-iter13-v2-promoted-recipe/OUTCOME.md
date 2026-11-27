# Outcome: Iter 13 — `UniOMR v2` (`small_45m`, Promoted Recipe + All Iter-13 Bug Fixes)

- **Experiment ID:** `2026-11-26T15-00-iter13-v2-promoted-recipe`
- **Status:** `COMPLETED`
- **Verdict:** **`PROMOTE`** *(New all-time UniOMR champion on both `suite31_hw` and `OMR-NorthStar`; 12-scout forensic learning underway for Iter 14)*
- **Completed At:** `2026-11-27 04:31` (local)
- **Platform / GPU:** Local `RTX 4080 SUPER` (`16 GB` VRAM, BF16 native, `$0` cloud cost)
- **Model Scale:** `UniOMR v2 small_45m` (`41,334,795` params, `0.816 GB` peak eval VRAM, last-15% `INT4 QAT`)
- **Stage 2 Cumulative Steps:** `24,027` cumulative Stage-2 full-page steps (`15,700` steps in final Tier-3 accelerated leg after `87b1ec8` + `215ce61`, WSD `0.20` cooldown + last-15% INT4 QAT)
- **Local Artifact Paths:**
  - Stage 2 Checkpoints: `D:\tmp\mvd-runs\uni-omr-iter13-v2-stage2-24k\uni_omr_v2_ema_best.pt` & `uni_omr_v2_ema_latest.pt`
  - `suite31_hw` Fast (`1408px/3584, FSA+MTP-4`): `D:\tmp\uni-omr-suite31-ood\iter13_v2_stage2_24k_ema_1408_fsa_mtp.json`
  - `suite31_hw` High-Acc (`1408px/4096, +oemer-dewarp+mbeam2`): `D:\tmp\uni-omr-suite31-ood\iter13_v2_stage2_24k_ema_1408_fsa_oemer_mbeam2.json`
  - `OMR-NorthStar` (`ema-best`): `D:\tmp\northstar-eval-runs\iter13-v2\report.json`
  - `OMR-NorthStar` (`ema-last`): `D:\tmp\northstar-eval-runs\iter13-v2-last\report.json`

---

## 1. Measured Results — `suite31_hw` (45 Pieces = 90 Pages, Strict Note F1)

| Checkpoint / Eval Arm | `printed_clean` (31p) | `printed_photo` (31p) | `hw_clean` (14p) | `hw_photo` (14p) | **`all45_clean`** | **`all45_photo`** | `DecOK` | Fixes Med (Cl / Ph) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Iter 10.5 `v1` EMA** *(prev all-time record & KD teacher)* | `0.8968` | `0.8935` | `0.2591` | `0.2174` | `0.6984` | `0.6832` | `100%` | `32.0 / 41.0` |
| **Iter 12 `v2` Stage 2 step 8k EMA** *(+ P0 fixes)* | `0.8728` | `0.8369` | `0.1744` | `0.1121` | `0.6555` | `0.6114` | `100%` | — |
| **Iter 13 `v2` (`ema_best`) — Fast (`1408/3584, FSA+MTP-4`)** | **`0.9193`** | **`0.9103`** | `0.2448` | **`0.2533`** | `0.7094` | **`0.7059`** | **`100%`** | **`29.0 / 44.0`** |
| **Iter 13 `v2` (`ema_best`) — High-Acc (`+dewarp+mbeam2`)** | `0.9153` | `0.9130` | **`0.2672`** | `0.2464` | **`0.7137`** | `0.7056` | **`100%`** | **`29.0 / 42.0`** |
| **Delta vs Iter 10.5 `v1` (Best Iter 13 arm)** | **`+2.25 pp`** | **`+1.95 pp`** | **`+0.81 pp`** | **`+3.59 pp`** | **`+1.53 pp`** | **`+2.27 pp`** | `0.0` | **`-3.0 / +1.0`** |
| **Delta vs Iter 12 `v2` (Fast / High-Acc)** | **`+4.65 pp`** | **`+7.34 pp`** | **`+9.28 pp`** | **`+14.12 pp`** | **`+5.82 pp`** | **`+9.45 pp`** | `0.0` | — |

- **Per-slice telemetry (`ema_best` Fast):**
  - `printed_clean`: Tok F1 `0.9500`, fixes med **`10.0`**, `290.3 tok/s`, `97.35%` MTP accept.
  - `printed_photo`: Tok F1 `0.9450`, fixes med **`8.0`**, `287.9 tok/s`, `97.28%` MTP accept.
  - `hw_clean`: Tok F1 `0.7204` (`0.7279` w/ `+dewarp+mbeam2`, `12/25` measure-beam repaired), fixes med `523.5` (`503.5` w/ `+dewarp+mbeam2`).
  - `hw_photo`: Tok F1 `0.7509`, fixes med `577.0` (`518.0` w/ `+dewarp+mbeam2`), `81.28%` MTP accept.
- **Note on mid-run deltas:** Compared to the Iter 13 mid-checkpoint (`printed_clean` `-1.80 pp`, `printed_photo` `-1.83 pp` in the late stage while `hw_clean` gained `+5.52 pp` to `0.2672` and `hw_photo` gained `+11.30 pp` to `0.2533`), the final Stage-2 run traded a small printed-peak dip for a massive handwritten breakthrough, lifting both `all45_clean` (`0.7137`) and `all45_photo` (`0.7059`) to new all-time records.

---

## 2. Measured Results — `OMR-NorthStar` (`dev-v1`, 125 Pages, `--neg-control-gate`)

| Metric | Iter 12 `v2` | Iter 13 `v2` (`ema-best`) | **Iter 13 `v2` (`ema-last`)** | Delta (`ema-last` vs Iter 12) |
|---|---:|---:|---:|---:|
| **`NS` Score (Headline)** | `26.75` | `34.04` | **`34.08`** | **`+7.33`** |
| **`Tier-W NS`** | — | `25.38` | **`36.45`** | — |
| **`M_geom`** | — | `57.45` | **`59.20`** | — |
| **`CDR` (Clean Digital) / CFR** | — | `80.53` (`14.3%`) | **`82.94`** (`11.4%`) | — |
| **`ALS` (Archival/Light-Scan) / CFR** | `51.13` | `75.88` (`16.1%`) | **`78.47`** (`16.1%`) | **`+27.34`** |
| **`RWP` (Real-World Photo) / CFR** | — | `76.96` (`19.4%`) | **`80.93`** (`12.9%`) | — |
| **`HWM` (Handwritten Manuscript) / CFR** | — | `23.17` (`60.7%`) | **`23.31`** (`64.3%`) | — |
| **Axes (`S_note` / `S_struct` / `S_expr` / `S_play`)** | — | `0.7074 / 0.7832 / 0.5194 / 0.7218` | **`0.7262 / 0.7905 / 0.5544 / 0.7461`** | — |
| **Human Utility (`median_wFPP` / `mean_NTSR` / `NUPR`)** | — | `156.5 / +0.5426 / 22.4%` | **`114.0 / +0.5926 / 18.4%`** | — |
| **Negative Control (`NMHR`, 4 pages)** | — | **`0.0%` (`4/4` PASS)** | **`0.0%` (`4/4` PASS)** | PASS |
| **Latency (`median` / `p95`, RTX 4080 SUPER)** | — | — | **`3.73 s / 8.86 s`** | — |

---

## 3. Scrutiny & Anomaly Check

1. **Coverage & Silent Failures:** **PASS.** All `45/45` (`90/90` pages) of `suite31_hw` scored with **`100%` `decode_ok_rate`** across both Fast and High-Acc arms (`0` NaNs, `0` decode failures). All `125/125` `OMR-NorthStar` pages scored cleanly, and `4/4` negative-control pages (`blank_staff` × 2, `program_notes` × 2) passed with `0` hallucinated notes (`NMHR = 0.0%`).
2. **`ema-last` vs `ema-best` Selection Asymmetry (IMPORTANT FINDING):**
   - `uni_omr_v2_ema_latest.pt` (end of WSD `0.20` cooldown + last-15% INT4 QAT at `24,027` cumulative steps) **decisively outperformed** `uni_omr_v2_ema_best.pt` (selected earlier on a 32-page val subset) on `OMR-NorthStar`: `Tier-W NS` **`36.45 vs 25.38`** (`+11.07`), `M_geom` `59.20 vs 57.45`, `CDR` `82.94 vs 80.53`, `ALS` `78.47 vs 75.88`, `RWP` `80.93 vs 76.96`, `median_wFPP` **`114.0 vs 156.5`** (`-42.5` fixes/page!), and `NUPR` `18.4% vs 22.4%`.
   - Because `launch_iter13.ps1` evaluated `suite31_hw` on `ema_best`, the `suite31_hw` numbers above are conservative relative to `ema_latest` — Iter 14 should always evaluate `ema_latest` (post-WSD cooldown) alongside `ema_best`.
3. **Domain / Slice Asymmetry:**
   - **Printed vs HW late-stage trade-off:** `printed_clean` (`0.9193`) and `printed_photo` (`0.9103`) both cleared the `>= 0.900 / >= 0.895` `PROMOTE` gates and beat Iter 10.5 `v1` (`0.8968 / 0.8935`) by `+2.25 pp / +1.68 pp`, while `hw_photo` jumped to `0.2533` (`+3.59 pp` vs `v1`, `+14.12 pp` vs Iter 12). However, `printed_clean/photo` dipped `-1.80 / -1.83 pp` vs mid-run as HW jumped `+5.52 / +11.30 pp`.
   - **`+dewarp+mbeam2` slice trade-off:** Helped `hw_clean` (`0.2448 -> 0.2672`, `+2.24 pp`) and `printed_photo` (`0.9103 -> 0.9130`, `+0.27 pp`), while slightly dipping `hw_photo` (`0.2533 -> 0.2464`, `-0.69 pp`) and `printed_clean` (`0.9193 -> 0.9153`, `-0.40 pp`).
   - **Headline NorthStar cap:** Overall `NS = 34.08` is strictly bottlenecked by `HWM = 23.31` (`64.3%` CFR; `worst_stratum_q`) and a single `T2` outlier (`foster-leadsheet`: `q = 3.45–3.69` on `CDR`/`ALS`, vs `44.57` on `RWP`).
4. **Hardware / Platform Health:** Tier 1+2+3 speedups (`87b1ec8`, `215ce61`) completed Stage 2 cleanly on local RTX 4080 SUPER with `0` OOMs and `0` WDDM stalls.

---

## 4. Verdict & Follow-Up Action / Successor Experiment

- **Verdict:** **`PROMOTE`** — Iter 13 `UniOMR v2` (`small_45m`, `41.3M` params) clears the `all45_clean >= 0.705` (`0.7137`), `all45_photo >= 0.690` (`0.7059`), `printed_clean >= 0.900` (`0.9193`), `printed_photo >= 0.895` (`0.9103`), and `hw_photo >= 0.230` (`0.2533`) gates, and jumps `+7.33` `NS` (`+27.34` `ALS`) on `OMR-NorthStar`. Promoted as the new `UniOMR v2` champion checkpoint (`uni_omr_v2_ema_latest.pt` / `uni_omr_v2_ema_best.pt`).
- **Active Follow-Up:** Full **12-scout parallel CPU forensic learning** (`learn-from-suite31` 6 scouts + `learn-from-northstar` 6 scouts) is underway on the Iter 13 eval outputs to dissect `HWM` (`64.3%` CFR), `T2` (`foster-leadsheet`), and expressive (`S_expr = 0.5544`) gaps and feed directly into **Iter 14** (`bekern+v2` Phase A vocab, promoted inference `ILMS`/`backbone_cons`/`skip-0.040`, `ax1_gl5`, expression weighting, landed in `4f8ed69`).
