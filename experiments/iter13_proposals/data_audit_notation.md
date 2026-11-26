# Iter 13 Data Variation Audit #2 — Musical Notation, Clef, Key, Meter, Polyphony & Genre Coverage

**Date:** 2026-11-26  
**Auditor:** Data Variation Auditor 2 (CPU-only)  
**Inputs Audited:**
- `D:\tmp\uni_fullpage_iter11_clean.jsonl` — `44,734` full-page rows (`36,490` `printed`, `7,775` synth `hw`, `469` `real_hw`):
  - **Unbiased 5,000-page uniform random sample** (`4,072` printed + `928` HW; `18,370` systems, `913,616` notes) for exact population rates.
  - **Full census of all `469` `real_hw` pages** (`1,363` systems, `130,206` notes; `452/469 = 96.4%` Debussy).
- `D:\tmp\real-hw-strips-iter7` — **Full census of all `6,213` strips** (`4,021` `hwsim`, `1,760` `deb`, `432` `moz`; `439,838` notes).
- `data-gen/score-corpus/suite31/suite31_hw_manifest.json` — **Full census of all `45` OOD pieces** (`31` printed + `14` real HW) parsed directly from MusicXML ground truth.
- `D:\tmp\uni-omr-suite31-ood\iter10_5_ema_best_suite31_hw_1536.json` — Iter 10.5 EMA best OOD evaluation (`0.8968` PR clean / `0.8935` PR photo; `0.2591` HW clean / `0.2174` HW photo Note F1).

---

## 1. Corpus Composition & Domain Imbalance

`uni_fullpage_iter11_clean.jsonl` (`44,734` rows) breaks down by source as:

| Source | Rows | Share | Domain |
|---|---:|---:|---|
| `corpus10x` | 12,702 | 28.4% | printed |
| `t32farm_mscz` | 10,445 | 23.3% | printed |
| `hw_sim_v04` | 6,700 | 15.0% | hw (synth) |
| `corpus100k` | 6,599 | 14.8% | printed |
| `verovio_iter10` | 2,991 | 6.7% | printed |
| `hw_sim_other` | 1,075 | 2.4% | hw (synth) |
| `uni-verovio-spacing-iter10` | 938 | 2.1% | printed |
| `uni-osmd-iter10` | 812 | 1.8% | printed |
| `uni-synth-1s2s-pages` | 761 | 1.7% | printed |
| `uni-printed-c-8vb-iter8` | 493 | 1.1% | printed |
| `real_hw_debussy` (+17 other `real_hw`) | 469 | 1.0% | **real_hw** |
| `uni-osmd-spacing-iter10` | 434 | 1.0% | printed |
| `uni-printed-rare-iter8` | 332 | 0.7% | printed |

**Headline structural finding:** `real_hw` is only **1.05%** of `iter11_clean` (`469 / 44,734`), and **96.4% of `real_hw` is a single composer (Debussy)** (`452 / 469` pages). Meanwhile, `iter11_hw` (dominated by synthetic `hw_sim_v04`) is **65.9% 3-staff** (Lieder) and only **22.0% 4-staff**, whereas `suite31_hw` is **57.1% 4-staff** (`8 / 14`: 5 string quartets + 2 choral/vocal + 1 two-piano).

---

## 2. Exact Notation Coverage Tables

All percentages are **page-level presence rates** (fraction of pages containing at least one instance; for `real-hw-strips-iter7`, strip-level presence rate).

### 2.1 Clef Distribution

| Clef | `iter11_all` (unif 5k) | `iter11_printed` (4,072) | `iter11_hw` (928) | `iter11_real_hw` (469) | `hw_strips_iter7` (6,213) | `suite31_printed` (31) | `suite31_hw` (14) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `clef-G-2` | 95.62% | 94.94% | 98.60% | 100.0% | 99.20% | 100.0% (31) | 100.0% (14) |
| `clef-F-4` | 75.82% | 70.78% | 97.95% | 99.1% | 91.31% | 64.5% (20) | 100.0% (14) |
| `clef-C-3` (alto) | 16.44% | 16.60% | 15.73% | **3.6%** (17) | 10.41% | 19.4% (6) | **35.7%** (5) |
| `clef-C-4` (tenor) | 5.22% | 5.77% | 2.80% | **1.3%** (6) | 2.45% | 0.0% (0) | 0.0% (0) |
| `clef-G-2-8vb` | 4.62% | 4.17% | 6.57% | **0.0%** (0) | **1.80%** | 9.7% (3) | **7.1%** (1) |
| `clef-C-1` (soprano) | 4.06% | 4.99% | 0.00% | 0.0% | 0.00% | 0.0% (0) | 0.0% (0) |
| `clef-p-2` (perc) | 2.12% | 2.60% | 0.00% | 0.0% | 0.00% | 0.0% (0) | 0.0% (0) |
| `clef-G-2-8va` | **0.12%** | **0.15%** (6) | **0.00%** (0) | **0.0%** (0) | **0.03%** (2) | **3.2%** (1) | 0.0% (0) |
| `mid_clef_change` | 11.90% | 13.04% | **6.90%** | 82.5% | 16.61% | 16.1% (5) | 21.4% (3) |

**Key clef gaps:**
- `clef-G-2-8va` is nearly extinct (`0.12%` overall, `0.00%` in `iter11_hw`, `0.03%` in HW strips) — directly implicated in `clara-polonaise-op1n1` (`cNF1=0.857`, 144 fixes).
- `clef-C-3` in `real_hw` is only **3.6%** (`17` pages) vs **35.7%** (`5/14`) in `suite31_hw` — all 5 real-HW string quartets (`hw-beethoven-op135`, `hw-mozart-k387`, `hw-mozart-k421`, `hw-schubert-d810`, `hw-mozart-k464`) average `0.148` Note F1.
- `clef-G-2-8vb` is **0.0%** in `real_hw` (`0/469`) and `1.80%` in HW strips — `hw-hensel-morgengruss-d-b` (SATB choral with `clef-G-2-8vb`) is the #1 worst HW piece (`cNF1=0.024`).

---

### 2.2 Key Signature Distribution (`-7 .. +7` & Mid-Piece Key Changes)

| Key | `iter11_all` (5k) | `iter11_printed` (4,072) | `iter11_hw` (928) | `iter11_real_hw` (469) | `hw_strips_iter7` (6,213) | `suite31_printed` (31) | `suite31_hw` (14) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `key-7` | **0.10%** | **0.12%** | **0.00%** | 0.6% | 0.14% | 0 | 0 |
| `key-6` | **0.82%** | **0.61%** | 1.72% | 2.6% | 1.34% | 0 | **14.3%** (2) |
| `key-5` | 2.40% | 1.92% | 4.53% | 7.9% | 4.75% | 0 | 0 |
| `key-4` | 6.12% | 5.70% | 7.97% | **1.9%** | 5.49% | 6.5% (2) | 0 |
| `key-3` | 8.88% | 8.30% | 11.42% | 7.5% | 10.35% | 9.7% (3) | 0 |
| `key-2` | 10.54% | 10.66% | 10.02% | **0.9%** (4) | 5.94% | 16.1% (5) | 7.1% (1) |
| `key-1` | 14.94% | 15.74% | 11.42% | 14.9% | 11.83% | 12.9% (4) | 35.7% (5) |
| `key+0` | 22.30% | 23.65% | 16.38% | 43.7% | 21.31% | 22.6% (7) | 0 |
| `key+1` | 16.82% | 17.76% | 12.72% | 15.4% | 10.19% | 19.4% (6) | 14.3% (2) |
| `key+2` | 12.52% | 13.26% | 9.27% | 14.7% | 9.03% | 6.5% (2) | 14.3% (2) |
| `key+3` | 7.46% | 7.15% | 8.84% | 13.2% | 7.52% | 16.1% (5) | 14.3% (2) |
| `key+4` | 3.92% | 3.59% | 5.39% | 14.5% | 7.00% | 6.5% (2) | 0 |
| `key+5` | 2.02% | 1.67% | 3.56% | 13.6% | 5.89% | 0 | 0 |
| `key+6` | **0.52%** | **0.39%** | 1.08% | 4.5% | 1.16% | 0 | 0 |
| `key+7` | **0.20%** | **0.20%** | **0.22%** | 1.9% | 0.55% | 0 | 0 |
| `mid_key_change` | **4.40%** | **5.08%** | **1.40%** | 17.3% | **2.48%** | **9.7%** (3) | 0% |
| `page_key_change` | 7.88% | 8.74% | 4.09% | 47.3% | 2.48% | 16.1% (5) | 0% |

**Key signature gaps:**
- Extreme keys (`|fifths| >= 5`) are only **4.9%** of `iter11_printed` (`key±6, ±7` combined = **1.32%**).
- `mid_key_change` is only **5.08%** of printed pages and **1.40%** of `iter11_hw` — yet ALL 3 printed Suite31 pieces with mid-page key changes (`beethoven-grosse-fuge` `0.788`, `handel-lascia` `0.821`, `clara-polonaise-op1n1` `0.857`) are in the **bottom 7 printed pieces**, with **98.5%–99.1% Token F1** but **78.8%–85.7% Note F1** (a single missed/misplaced `key±N` token corrupts accidental spelling across every subsequent bar!).

---

### 2.3 Time Signature Distribution & Pickups

> **VOCAB note:** `data.VOCAB` has **no distinct `time-C` or `time-cut-C` tokens** — common time `C` is collapsed into `time-4-4` and cut time `¢` into `time-2-2`. Visually, however, `C` and `¢` glyphs are distinct from `4/4` and `2/2` numerals (`9/45` Suite31 pieces use `C`, `1/45` uses `¢`).

| Meter | `iter11_all` (5k) | `iter11_printed` (4,072) | `iter11_hw` (928) | `iter11_real_hw` (469) | `hw_strips_iter7` (6,213) | `suite31_printed` (31) | `suite31_hw` (14) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `time-4-4` (incl. `C`) | 38.24% | 38.70% | 36.21% | 41.4% | 33.00% | 32.3% (5×`4/4`, 5×`C`) | 28.6% (4×`C`) |
| `time-3-4` | 18.20% | 17.98% | 19.18% | 29.4% | 21.20% | 22.6% (7) | 21.4% (3) |
| `time-2-4` | 14.26% | 13.43% | 17.89% | 24.7% | 18.88% | 19.4% (6) | 21.4% (3) |
| `time-6-8` | 14.16% | 14.49% | 12.72% | 11.9% | 11.85% | 12.9% (4) | 21.4% (3) |
| `time-2-2` (incl. `¢`) | 8.06% | 8.74% | 5.06% | 4.7% | 5.55% | 9.7% (2×`2/2`, 1×`¢`) | 7.1% (1) |
| `time-3-2` | **3.26%** | **3.46%** | **2.37%** | 6.0% | **1.61%** | 3.2% (1) | 0 |
| `time-4-2` | **2.92%** | **3.49%** | **0.43%** | **0.6%** | **0.06%** | 6.5% (2) | 0 |
| `time-3-8` | 2.90% | 2.41% | 5.06% | 19.0% | 6.62% | 3.2% (1) | 0 |
| `time-9-8` | **2.52%** | **2.53%** | **2.48%** | 1.3% | **1.93%** | 0 | 0 |
| `time-7-8` | **2.06%** | **2.50%** | **0.11%** | **0.0%** | **0.02%** | 0 | 0 |
| `time-12-8` | **1.36%** | **1.35%** | **1.40%** | 3.8% | **1.80%** | 0 | 0 |
| `time-5-4` | **0.52%** | **0.59%** | **0.22%** | **0.0%** | **0.08%** | 0 | 0 |
| `time-1-16` / `1-32` | **<0.35%** | **0.32%** | **0.00%** | **0.0%** | **0.00%** | 6.5% (2) | 0 |
| `mid_time_change` | 10.92% | 12.48% | **4.09%** | 20.7% | **4.62%** | 16.1% (5) | 0 |
| `pickup_anacrusis` | 23.98% | 22.96% | 28.45% | **5.8%** | **12.51%** | **35.5%** (11) | **28.6%** (4) |

**Meter gaps:**
- `time-3-2` and `time-4-2` (half-note beat meters, especially paired with 1-staff folk pickups) cause **-0.209 Note F1** on Suite31 Printed (`essen-ballad` `0.515`, `essen-altdeu` `0.610`).
- Asymmetric/compound meters `5/4` (`0.52%`), `12/8` (`1.36%`), `7/8` (`2.06%`, `0.11%` HW), `9/8` (`2.52%`) are severely underrepresented, especially in HW.
- `pickup_anacrusis` in `real_hw` is only **5.8%** (`27/469`) vs **28.6%** (`4/14`) in `suite31_hw` and **35.5%** (`11/31`) in `suite31_printed`.

---

### 2.4 Staves/System, Systems/Page, Polyphonic Voices & Expressive Features

| Dimension | `iter11_all` (5k) | `iter11_printed` (4,072) | `iter11_hw` (928) | `iter11_real_hw` (469) | `hw_strips_iter7` (6,213) | `suite31_printed` (31) | `suite31_hw` (14) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Staves = 1** | 20.3% | 24.8% | **0.3%** | **0.0%** | **0.1%** | 29.0% (9) | 0% |
| **Staves = 2** | 17.8% | 20.5% | **5.8%** | 45.2% | 17.1% | 19.4% (6) | 14.3% (2) |
| **Staves = 3** | 30.8% | 22.8% | **65.9%** | 24.5% | **63.2%** | 16.1% (5) | 28.6% (4) |
| **Staves = 4** | 17.4% | 16.4% | **22.0%** | 30.3% | **19.7%** | 25.8% (8) | **57.1%** (8) |
| **Staves = 5** | 4.5% | 4.7% | 3.7% | 0.0% | 0.0% | 3.2% (1) | 0% |
| **Staves >= 6** | 9.2% | 10.7% | 2.3% | 0.0% | 0.0% | 6.5% (2) | 0% |
| **Systems = 1..2** | 28.4% | 30.3% | 20.1% | 48.6% | — | 16.1% (5) | 14.3% (2) |
| **Systems = 3..5** | 57.4% | 55.1% | 67.8% | 46.6% | — | 71.0% (22) | 78.6% (11) |
| **Systems = 6..7** | 9.5% | 9.0% | 11.4% | 4.0% | — | 12.9% (4) | 7.1% (1) |
| **Systems >= 8** | 4.7% | 5.6% | 0.6% | 0.6% | — | 0% | 0% |
| **Voices = 1..2** | 26.6% | 32.5% | 0.8% | — | 2.7% | 29.0% (9) | 0% |
| **Voices = 3..4** | 42.5% | 38.9% | 58.4% | — | 77.3% | 38.7% (12) | 57.1% (8) |
| **Voices = 5..6** | 22.8% | 19.6% | 36.8% | — | 19.1% | 25.8% (8) | 42.9% (6) |
| **Voices >= 7** | 8.1% | 9.0% | 4.0% | — | 0.9% | 6.5% (2) | 0% |
| **Tuplets** | 19.30% | 17.26% | 28.23% | 71.4% | 25.27% | 32.3% (10) | 35.7% (5) |
| **Grace notes** | 16.82% | 14.66% | 26.29% | 34.8% | 16.82% | 35.5% (11) | 28.6% (4) |
| **Dense chord >=3** | 39.38% | 33.13% | 66.81% | 80.4% | 63.24% | 38.7% (12) | 57.1% (8) |
| **Dense chord >=4** | 17.16% | 13.78% | 32.00% | 45.0% | 36.71% | 19.4% (6) | 35.7% (5) |
| **Repeats (`bar-repeat-*`)** | 20.62% | 25.29% | **0.11%** (1/928) | **1.9%** | **0.84%** | 38.7% (12) | 7.1% (1) |
| **Volta / 1st-2nd endings** | **0.00%** (no VOCAB) | **0.00%** | **0.00%** | **0.00%** | **0.00%** | **9.7%** (3) | 0% |
| **Extreme ledger (>=3 lines)** | 51.20% (4.78% notes) | 45.97% (4.24%) | 74.14% (6.28%) | 96.4% (13.76%) | 65.88% (9.50%) | 45.2% (14) | 64.3% (9) |

---

## 3. Cross-Reference With Iter 10.5's Worst Printed & HW Pieces

### 3.1 Worst 9 Printed Pieces (`iter10_5_ema_best_suite31_hw_1536.json`)

| Piece | Clean NF1 | Photo NF1 | Clean TokF1 | HR | Fixes | Staves / Sys / Voices | Exact Notation Features | Root Cause Diagnosis |
|---|---:|---:|---:|---:|---:|---|---|---|
| `foster-leadsheet` | **0.021** | 0.021 | 0.052 | 0.01 | 94 | 1 / 2 / 1 | `clef-G-2`, `key0`, `4/4`, `rep`, **`volta`**, chord symbols | Lead-sheet layout + volta bracket (`0` VOCAB tokens) -> near-total truncation (`hypN=1/95`). |
| `essen-ballad` | **0.515** | 0.944 | **0.975** | 0.83 | 32 | 1 / 2 / 1 | `clef-G-2`, `key+2`, **`time-3-2`**, `pickup` | Rare `3/2` meter + 1-staff pickup -> barline/beat shift (97.5% TokF1 vs 51.5% NoteF1!). |
| `essen-altdeu` | **0.610** | 0.438 | **0.983** | 0.97 | 46 | 1 / 5 / 1 | `clef-G-2`, `key+1`, **`time-4-2`**, `pickup` | Rare `4/2` meter + 1-staff pickup -> beat/bar drift (98.3% TokF1 vs 61.0% NoteF1!). |
| `beethoven-grosse-fuge` | **0.788** | 0.808 | **0.988** | 1.03 | 62 | 4 / 3 / 4 | `C-3,F-4,G-2`, **`key+1->-1`**, **`2/4->6/8`**, `grace` | **`midKey` + `midTime`** in 4-staff SQ: 98.8% TokF1 but 78.8% NoteF1 from key/meter switch drift. |
| `handel-lascia` | **0.821** | 0.821 | **0.991** | 1.02 | 59 | 3 / 4 / 5 | `F-4,G-2`, **`key+2->+3`**, **`3/4->C`**, **`midClef`** | **Triple mid-piece change** (`midClef` + `midKey` + `midTime`): 99.1% TokF1 -> 82.1% NoteF1! |
| `cpebach-h186` | **0.839** | 0.827 | 0.982 | 1.02 | 93 | 2 / 6 / 5 | `F-4,G-2`, `key+3`, `C`, `tup`, `grace` | Dense 5-voice Baroque keyboard polyphony on 2 staves + ornaments/tuplets. |
| `clara-polonaise-op1n1` | **0.857** | 0.838 | 0.985 | 1.00 | 144 | 2 / 6 / 5 | `F-4,G-2`, **`clef-G-2-8va`**, **`key-3->-4`**, `midClef`, `94` ext ledger | **`clef-G-2-8va`** (`0.12%` in train!) + `midKey` + extreme ledger-line piano chords (`ch=4`). |
| `oneills-tune` | **0.868** | 0.831 | 0.969 | 1.03 | 32 | 1 / 4 / 1 | `G-2`, `key-2`, **`1/16 -> 2/4`**, `grace` | Exotic pickup encoded as **`time-1-16`** (`<0.3%` in train) + `midTime` + folk grace notes. |
| `ryans-jig` | **0.883** | 0.883 | 0.990 | 1.01 | 27 | 1 / 3 / 1 | `G-2`, `key-2`, `2/4`, `tup`, `grace`, `rep`, `pickup` | Dense 1-staff folk grace notes + tuplets + repeat pickup. |

**Feature-vs-Note-F1 correlations on Suite31 Printed (`n=31`):**
- `has_volta`: **-0.253** (`0.668` vs `0.921`)
- `time-3-2 / 4-2`: **-0.209** (`0.708` vs `0.917`)
- `staves == 1`: **-0.186** (`0.765` vs `0.951`) — 1-staff folk/leadsheet pieces dominate the printed tail!
- `mid_key_change`: **-0.083** (`0.822` vs `0.905`)
- `clef-G-2-8va`: **-0.041** (`0.857` vs `0.898`, 144 fixes)

### 3.2 Worst 10 Handwritten Pieces (`iter10_5_ema_best_suite31_hw_1536.json`)

| Piece | Clean NF1 | Photo NF1 | Clean TokF1 | HR | Fixes | Staves / Sys | Exact Notation Features | Root Cause Diagnosis |
|---|---:|---:|---:|---:|---:|---|---|---|
| `hw-hensel-morgengruss-d-b` | **0.024** | 0.025 | 0.652 | 0.64 | 239 | **4** / 2 | `F-4,G-2`, **`G-2-8vb`**, `key+2`, `6/8`, `pickup` | **4-staff SATB choral + `clef-G-2-8vb`** (`0.0%` in `real_hw`!) + pickup. |
| `hw-mendelssohn-herbstlied-d-b` | **0.070** | 0.067 | 0.694 | **1.76** | 924 | **4** / 5 | `F-4,G-2`, `key+3`, `6/8`, `ch=3` | **4-staff vocal+piano** (not Debussy) -> massive over-generation (`HR=1.76`). |
| `hw-beethoven-op135` | **0.080** | 0.056 | 0.695 | **1.56** | 254 | **4** / 2 | **`clef-C-3`**, `F-4,G-2`, `key-1`, `2/4`, `grace` | **4-staff String Quartet (`clef-C-3`)** — only `3.6%` of `real_hw` has `C-3`! |
| `hw-mozart-k387` | **0.083** | 0.066 | 0.740 | **1.44** | 967 | **4** / 3 | **`clef-C-3`**, `F-4,G-2`, `key+1`, `C`, `grace`, `pickup` | **4-staff String Quartet (`clef-C-3`)** + `C` glyph + pickup + grace. |
| `hw-hensel-im-herbste-d-b` | **0.099** | 0.108 | 0.612 | **0.49** | 474 | 3 / 4 | `F-4,G-2`, `key-2`, `C`, `v=5`, `ch=3` | Non-Debussy German Lied (`key-2` is `0.9%` of `real_hw`!) -> under-generation (`HR=0.49`). |
| `hw-mozart-k421` | **0.152** | 0.074 | 0.668 | **0.58** | 1,093 | **4** / 3 | **`clef-C-3`**, `F-4,G-2`, `key-1`, `C`, `tup`, `grace` | **4-staff String Quartet (`clef-C-3`)** + dense tuplets/grace. |
| `hw-schubert-d810` | **0.181** | 0.108 | 0.673 | 0.72 | 534 | **4** / 4 | **`clef-C-3`**, `F-4,G-2`, `key-1`, `C`, `tup` | **4-staff String Quartet (`clef-C-3`)**. |
| `hw-cornelius-botschaft-d-b` | **0.240** | 0.341 | 0.823 | 0.85 | 330 | 3 / 4 | `F-4,G-2`, `key+2`, `6/8`, `midClef`, `ch=4` | Non-Debussy Lied with `midClef` + 4-note chords. |
| `hw-mozart-k464` | **0.242** | 0.059 | 0.841 | 0.72 | 540 | **4** / 3 | **`clef-C-3`**, `F-4,G-2`, `key+3`, `3/4`, `pickup` | **4-staff String Quartet (`clef-C-3`)** + pickup. |
| `hw-debussy-lindaraja` | **0.319** | 0.286 | 0.798 | 1.11 | 1,078 | **4** / 4 | `F-4,G-2`, `key-1`, `2/4`, `midClef`, `tup`, `grace`, `30` ledg | 4-staff 2-piano extreme density (`749` ref notes, `30` ext ledger). |

**Feature-vs-Note-F1 correlations on Suite31 HW (`n=14`):**
- `staves == 4` (8/14 pieces): **-0.269** (`0.144` vs `0.413`)!
- `clef-G-2-8vb` (1/14): **-0.253** (`0.024` vs `0.277`)
- `non-debussy HW` (10/14): **-0.219** (`0.196` vs `0.416`)
- `clef-C-3` string quartets (5/14): **-0.173** (`0.148` vs `0.321`)
- `time-6-8` (3/14): **-0.188** (`0.112` vs `0.299`)
- `time-C (4/4)` (4/14): **-0.182** (`0.129` vs `0.311`)

---

## 4. Concrete Iter 13 Synthetic & Corpus Upsampling Recipes

> **Implementation note:** `data-gen/notation-vary` is no longer present on disk; the active rendering stack is `tmp-train/mvd/render_verovio_dataset.py` (5 Verovio SMuFL fonts + exact SVG `<g class="system">`/`<g class="measure">` alignment), `tmp-train/mvd/multi_engrave.py` (Verovio + LilyPond), `data-gen/score-render` (MuseScore batch), and `hw-score-sim` (`C:\Users\grego\projects\hw-score-sim`). All synthetic MusicXML transforms below should be implemented as a deterministic Python pre-render mutator (`tmp-train/mvd/mutate_musicxml_iter13.py`) fed directly into `render_verovio_dataset.py` (printed) and `hw-score-sim` (synthetic HW), enforcing `build_strict_eval_filter()` (0 eval overlap, 0 NC).

### Recipe 1 — HW 4-Staff String Quartet & SATB Choral Pack (`+3,000` `hw-score-sim` pages)
- **Target gap:** Fixes the `-0.269` `S=4` HW collapse, `-0.173` `clef-C-3` HW gap, and `-0.253` `clef-G-2-8vb` HW gap (`real_hw` is 96.4% Debussy with only 3.6% `C-3` and 0% `G-2-8vb`).
- **Source MusicXML:** Permissive OpenScore String Quartets (excluding `k458`, `k387`, `k421`, `k464`, `op135`, `d810`, `op18n1`, `op41n1`, `op74n1`, `k155`, `grosse-fuge`) + PDMX 4-staff SATB choral & chamber scores (`S=4`, `clef-C-3` viola, `clef-G-2-8vb` tenor, `clef-C-4` cello).
- **Augmentations:**
  - Force `C` (`symbol="common"`) and `¢` (`symbol="cut"`) time-signature glyphs on 50% of `4/4` and `2/2` scores.
  - Inject 1st-bar pickups (`implicit="yes"`, 1–3 beats) on 30% of pieces, and `bar-repeat-left`/`bar-repeat-right` on 25% of pieces (raising `iter11_hw` repeat rate from `0.11%` to `>10%`).
  - Render via `hw-score-sim` with varied historical ink/paper styles (German/Classical manuscript spacing, 2–5 systems/page) to break Debussy-only HW overfitting.
- **Target share in Iter 13 HW:** Raise `S=4` in `iter11_hw` from `22.0% -> 40%`, `clef-C-3` from `15.7% -> 28%`, `clef-G-2-8vb` from `6.6% -> 12%`, `repeat` from `0.11% -> 12%`.

### Recipe 2 — Mid-Piece Key + Meter + Clef Change Splicer (`+2,500` printed Verovio/LilyPond + `+1,000` HW pages)
- **Target gap:** Fixes the `99% TokF1 -> 79–85% NoteF1` cliff on `beethoven-grosse-fuge`, `handel-lascia`, and `clara-polonaise-op1n1` (`mid_key_change` is only `5.08%` printed / `1.40%` HW).
- **Transform (`mutate_musicxml_iter13.py --mode mid-change`):**
  - On clean 1–4 staff PDMX/OpenScore MusicXMLs, pick 1–2 interior barlines per page (never bar 1) and insert:
    - **Key change** (`±1..±4` fifths, including natural-cancel key signatures, transposing all subsequent pitches diatonically + chromatically so engraving & bekern stay 100% consistent) — 60% of pages.
    - **Meter change** (e.g., `3/4 <-> C`, `2/4 <-> 6/8`, `4/4 <-> 3/2`, `6/8 <-> 2/2`) — 50% of pages.
    - **Mid-bar / mid-system clef change** (`F-4 <-> G-2`, `F-4 <-> C-4`, `G-2 <-> G-2-8va`) — 40% of pages.
  - Render 70% through `render_verovio_dataset.py` (all 5 SMuFL fonts) + 30% through `multi_engrave.py` (LilyPond) + `1,000` through `hw-score-sim`.
- **Target share:** Raise `mid_key_change` from `4.4% -> >=12%` overall (`>=8%` HW) and `mid_time_change` in HW from `4.1% -> >=10%`.

### Recipe 3 — Archaic, Compound & Asymmetric Meter + 1-Staff Folk/Lead-Sheet Pack (`+2,000` printed + `+500` HW pages)
- **Target gap:** Fixes `-0.209` Note F1 on `3/2`/`4/2` (`essen-ballad`, `essen-altdeu`), `-0.186` on `staves==1` (`foster-leadsheet`, `oneills-tune`, `ryans-jig`), and `<2%` coverage of `5/4`, `7/8`, `12/8`, `9/8`.
- **Transform (`mutate_musicxml_iter13.py --mode meter-resample`):**
  - **Double-value re-metering:** Convert `3/4 -> 3/2` and `4/4 -> 4/2` (doubling all note/rest durations & types `quarter->half`, `eighth->quarter`, etc.) on 1-staff folk melodies and 2–4 staff chorales/hymns, preserving pickups (`implicit="yes"`) and adding `time-1-16`/`time-1-32` short anacrusis bars where appropriate.
  - **Asymmetric/compound upsampling:** Upsample existing PDMX `5/4`, `7/8`, `9/8`, `12/8` scores across all 5 Verovio fonts + LilyPond + `hw-score-sim`.
  - **1-staff Lead-Sheet & Volta Pack:** Render `500` 1-staff PDMX lead sheets **with chord symbols (`<harmony>`) and 1st/2nd volta brackets (`<ending>`) visible in the image** (while keeping bekern GT clean, or adding `volta-1`/`volta-2` tokens if VOCAB is expanded) so the encoder learns to ignore chord-symbol text above 1-staff melodies instead of collapsing (`foster-leadsheet` `HR=0.01`!).
- **Target share:** Raise `time-3-2` and `time-4-2` to `>=5%` each (with 40% on 1-staff + pickup), `time-5-4` from `0.52% -> 2.5%`, `time-12-8` from `1.36% -> 3.0%`, `time-7-8` from `2.06% -> 3.0%` (`>=1.5%` HW).

### Recipe 4 — Rare Clef & Extreme Key Signature Transposition Pack (`+1,500` printed + `+500` HW pages)
- **Target gap:** Fixes `clef-G-2-8va` (`0.12%` printed, `0.00%` HW), `clef-C-4` (`2.80%` HW), `clef-G-2-8vb` (`4.17%` printed), and extreme keys `±5, ±6, ±7` (`1.32%` for `±6,±7` in printed).
- **Transform (`mutate_musicxml_iter13.py --mode transpose-reclef`):**
  - **Diacritic/key transposition:** Transpose clean 1–4 staff scores to `key-7, -6, -5, +5, +6, +7` (strictly keeping enharmonic spelling valid via `music21` or diatonic-fifth shift, verified by `bekern-decode` + `PrefixState`).
  - **Re-cleffing:** Re-clef high treble piano/violin passages to `clef-G-2-8va` (shifting written octave down 1 while preserving sounding pitch — directly matching `clara-polonaise-op1n1`!), cello/bassoon passages to `clef-C-4`, and tenor vocal lines to `clef-G-2-8vb`.
- **Target share:** Raise `clef-G-2-8va` from `0.12% -> >=2.5%` (printed + HW), `key±6/±7` from `1.64% -> >=5.0%`.

### Recipe 5 — Real-HW Corpus Diversification (Non-Debussy Manuscripts)
- **Target gap:** `real_hw` is 96.4% Debussy (`452/469` pages), creating a `-0.219` Note F1 generalization cliff to non-Debussy manuscripts (`0.416` Debussy vs `0.196` non-Debussy on `suite31_hw`).
- **Action:** Prioritize harvesting & aligning non-Debussy permissive historical manuscripts (e.g., via `harvest_openscore_sq_hw.py`, `harvest_bach_chopin_hw.py`, `harvest_pdm_openscore.py`, excluding all 14 `suite31_hw` works + `mozart-k458`), specifically targeting **4-staff string quartets (`clef-C-3`)**, **German Lieder (`key-2`, `6/8`, `C`)**, and **SATB choral (`clef-G-2-8vb`)**. Even `+150` non-Debussy aligned pages (upsampled 3× in the HW mix) will cut Debussy's `real_hw` share from `96.4%` down to `<50%`.

---

## 5. Verification Checklist for Iter 13 Manifest Build

Before launching Iter 13 training, re-run this audit script on `uni_fullpage_iter13.jsonl` and assert:
1. `clef-G-2-8va >= 2.0%` overall (`>= 1.5%` HW)
2. `mid_key_change >= 10.0%` printed, `>= 6.0%` HW
3. `iter11_hw` -> `iter13_hw`: `staves == 4 >= 38%`, `clef-C-3 >= 25%`, `clef-G-2-8vb >= 10%`, `repeat >= 10%`
4. `time-3-2 >= 4.5%`, `time-4-2 >= 4.5%`, `time-5-4 >= 2.0%`, `time-12-8 >= 2.5%`
5. `|fifths| >= 6 >= 4.0%` overall
6. `0` OOV tokens, `0` eval/suite31/k458/NC overlap, and `100%` `bekern-decode` + `PrefixState` pass rate.
