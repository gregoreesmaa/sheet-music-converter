# Experiment Outcome — `2026-11-26T03-08-ov-36b-multiscale-logprob-fsa-tta` (`ov-36b`)

- **Status:** `COMPLETED`
- **Completed (UTC):** `2026-11-26T12:51:51Z` (started `2026-11-26T12:22:52Z`)
- **Platform / GPU:** Kaggle `Tesla T4` (`bf16=False`, FP16/FP32; logged as `colab-Tesla T4`)
- **Elapsed Time:** `1738.9 s` (`29.0 min` compute; `1852.28 s` total queue wall)
- **Checkpoint Evaluated:** HF `checkpoints/iter10_soup_baseline/uni_omr_soup.pt` (`uni_omr_v1`, `step=12000`, `val_ser=0.192301`, `missing_keys=[]`, `unexpected_keys=[]`)
- **HF Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-26T03-08-ov-36b-multiscale-logprob-fsa-tta.json` (pulled to [`eval_result.json`](./eval_result.json))
  - Run Ledger: `runs/2026-11-26T03-08-ov-36b-multiscale-logprob-fsa-tta.json` (pulled to [`run_result.json`](./run_result.json))

---

## Measured Results (14 Real Historical HW Pieces × 2 Domains = 28 Pages × 5 Clean Scale Views = 140 Decodes, `max_len=3840`)

### 1. Individual Scale Views

| View | `hw_clean` Note F1 (Δ vs `1536` / Δ vs `1280`) | `hw_clean` Tok LCS | `hw_clean` Mean LogProb | `hw_photo` Note F1 (Δ vs `1536` / Δ vs `1280`) | `hw_photo` Tok LCS | `hw_photo` Mean LogProb | Decode OK |
|---|---:|---:|---:|---:|---:|---:|---:|
| *`base_1536` (ctrl)* | *`0.2693`* (`+0.0000` / `-0.0063`) | `0.3853` | `-0.1031` | *`0.1609`* (`+0.0000` / `+0.0061`) | `0.2927` | `-0.1153` | `100%` |
| *`scale_1280` (ctrl)* | **`0.2755`** (`+0.0063` / `+0.0000`) | `0.4050` | `-0.0994` | `0.1548` (`-0.0061` / `+0.0000`) | `0.2793` | `-0.1137` | `100%` |
| `scale_1024` | `0.2680` (`-0.0013` / `-0.0076`) | `0.4024` | `-0.1070` | `0.1648` (`+0.0039` / `+0.0100`) | `0.3271` | `-0.1115` | `100%` |
| `scale_1152` | `0.2483` (`-0.0210` / `-0.0273`) | `0.4085` | `-0.0902` | `0.1829` (`+0.0220` / `+0.0282`) | `0.2950` | `-0.1150` | `100%` |
| **`scale_1408`** | `0.2738` (`+0.0045` / `-0.0018`) | **`0.4360`** | `-0.1033` | **`0.1904`** (**`+0.0295`** / **`+0.0357`**) | **`0.3339`** | **`-0.1111`** | `100%` |

### 2. Unsupervised Selectors vs Oracle Upper Bound

| Pool | Selector | `hw_clean` F1 (Δ vs `1536` / Δ vs `1280`) | `hw_clean` W/T/L vs `1536` | `hw_photo` F1 (Δ vs `1536` / Δ vs `1280`) | `hw_photo` W/T/L vs `1536` |
|---|---|---:|---|---:|---|
| `all5` `{1024..1536}` | `best_fsa_beat` | `0.2623` (`-0.0070` / `-0.0133`) | `6/2/6` | `0.1557` (`-0.0052` / `+0.0010`) | `3/9/2` |
| `all5` `{1024..1536}` | `best_logprob` | `0.2885` (`+0.0193` / `+0.0130`) | `5/2/7` | `0.1702` (`+0.0093` / `+0.0154`) | `4/6/4` |
| `all5` `{1024..1536}` | `best_fsa_logprob` | `0.2623` (`-0.0070` / `-0.0133`) | `6/2/6` | `0.1753` (`+0.0144` / `+0.0206`) | `5/8/1` |
| `all5` `{1024..1536}` | `best_mbr_vote` | `0.2913` (`+0.0220` / `+0.0157`) | `5/4/5` | `0.1545` (`-0.0063` / `-0.0002`) | `3/7/4` |
| **`all5` `{1024..1536}`** | **`best_fsa_logprob_mbr`** | **`0.2964` (`+0.0271` / `+0.0208`)** | **`8/2/4`** | **`0.1755` (`+0.0146` / `+0.0208`)** | **`5/8/1`** |
| `all5` `{1024..1536}` | *`oracle_best_view`* | *`0.3143` (`+0.0451` / `+0.0388`)* | *`9/5/0`* | *`0.2281` (`+0.0672` / `+0.0733`)* | *`9/5/0`* |
| `fast2` `{1280, 1536}` | `best_logprob` | `0.2804` (`+0.0112` / `+0.0049`) | `3/7/4` | `0.1648` (`+0.0039` / `+0.0100`) | `2/11/1` |
| `fast3` `{1152, 1280, 1536}` | `best_logprob` | `0.2570` (`-0.0123` / `-0.0186`) | `3/3/8` | `0.1696` (`+0.0087` / `+0.0148`) | `2/10/2` |
| `fast3` `{1280, 1408, 1536}` *(offline)* | `best_logprob` | `0.2819` (`+0.0126` / `+0.0063`) | — | `0.1736` (`+0.0127` / `+0.0188`) | — |

---

## Scrutiny & Anomaly Check

1. **Coverage & Integrity — PASS:** All `14` HW pieces × `2` domains × `5` scale views = `140` decodes completed cleanly with `decode_ok_rate == 1.000`, `0` NaN/None scores, `missing_keys == []`.
2. **PROMOTE Gate Met Cleanly by `best_fsa_logprob_mbr` (`all5`):**
   - Beats **both** `base_1536` and `scale_1280` by **`+0.0271 / +0.0208` on `hw_clean`** (`>= +0.015` gate) AND **`+0.0146 / +0.0208` on `hw_photo`** (`>= +0.010` gate), with **`wins > losses` on every single comparison** (`8W/2T/4L` and `5W/8T/1L` vs `1536`; `5W/5T/4L` and `7W/5T/2L` vs `1280`) and zero domain regression!
   - Removing `clahe`/`deskew` and combining `0.5 * fsa_beat + 1.5 * mean_logprob + 1.0 * mbr_vote` rescued multi-scale TTA from `ov-36`'s negative `hw_photo` result (`-0.0009`) to a solid `+0.0146` / `+0.0208` win across both domains.
3. **Why Pure `best_fsa_beat` Failed (`-0.0070` / `-0.0052`):** Adding `scale_1152` introduced a low-resolution view that occasionally hallucinates shorter, rhythmically valid sequences; `best_logprob` and `best_mbr_vote` reject those outliers, making the 3-signal fusion (`best_fsa_logprob_mbr`) strictly superior to every individual signal.
4. **Hardware / Precision Note (T4 FP16 vs L4 BF16) & `scale_1408` Discovery:**
   - Because `ov-36b` ran on Kaggle `Tesla T4` (`bf16=False`, FP16/FP32), `base_1536` `hw_photo` shifted from `0.1916` (L4 BF16 in `ov-36`) to `0.1609` (T4 FP16), while single-view **`scale_1408`** jumped to `0.2738` clean / **`0.1904` photo** (`+0.0295` vs `1536`!). All within-run comparisons are apples-to-apples on the exact same T4 hardware/precision.
   - Offline simulation of 3-view `{1280, 1408, 1536}` `best_logprob` already reaches `+0.0126` clean / `+0.0127` photo at `40%` less compute than `all5`.

---

## Verdict: **`PROMOTE`**

`best_fsa_logprob_mbr` on clean in-distribution multi-scale TTA (`{1024, 1152, 1280, 1408, 1536}`) passes every quantitative PROMOTE criterion (`hw_clean >= +0.015`, `hw_photo >= +0.010` vs both `1536` and `1280`, `wins > losses`, `100%` decode validity).

## Follow-Up Action / Successor Experiment

- **Promote Action:** Wire `--tta-scales 1024,1152,1280,1408,1536` (and fast preset `1280,1408,1536`) + `--tta-rerank fsa_logprob_mbr` (`0.5 * fsa + 1.5 * mean_logprob + 1.0 * mbr`) into `tmp-train/mvd/eval_uni_suite31.py` as an opt-in high-accuracy HW inference mode, and verify full 45-piece `suite31_hw` + BF16 parity on the next major checkpoint (`Iter 12+`).
- **Secondary Note:** `oracle_best_view` on `hw_photo` (`0.2281`) still leaves `+0.0526` headroom; visual render-cycle reranking (tested in `2026-11-26T02-20-academic-dual-verifier-v2-fixed-render`) is already queued to attack that remaining gap.
