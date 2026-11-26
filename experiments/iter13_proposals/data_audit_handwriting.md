# Iter 13 Data Audit 3 — Handwritten Manuscript, Scribe-Style & Sim-to-Real Diversity

**Date:** 2026-11-26  
**Scope:** Audited `D:\tmp\uni_fullpage_iter11_clean.jsonl` (`real_hw`: 469 rows; `hw`: 7,775 rows), `C:\Users\grego\projects\hw-score-sim` (`v04`/`v05`/`v06`), `D:\tmp\unlabeled-10k-real\rism_hw` (5,707 pages), idle `D:\tmp` PDM/NIFC real HW corpora (~1,728 pages), `CUT-RISM` (Exp 1), and promoted `ov-39` RISM pseudo-labels against the 14 historical HW pieces in `D:\tmp\uni-omr-suite31-ood\iter10_5_ema_best_suite31_hw_1536.json`.

---

## 1. Executive Summary: Why `suite31_hw` HW Note F1 Is 0.2591 Clean / 0.2174 Photo

The 3×–5× performance gap across `suite31_hw`'s handwritten strata maps **1-to-1** to training data skew:

| `suite31_hw` Stratum | Pieces | `iter10_5` Clean Note F1 | `iter10_5` Photo Note F1 | Clean Tok F1 | Share of `iter11` `real_hw` |
|---|---:|---:|---:|---:|---:|
| **`hw_portrait_debussy`** (Debussy, c.1890–1918, G2/F4, 0 lyrics) | 4 | **0.4156** | **0.3782** | **0.8610** | **96.4% (452 / 469)** |
| **`hw_portrait_pdm`** (Beethoven/Cornelius/Hensel/Mendelssohn Lieder, 32–276 German cursive lyrics) | 5 | **0.2454** | **0.2335** | **0.7268** | **0.0% (0 / 469)** |
| **`hw_landscape`** (Mozart/Beethoven/Schubert 4-staff String Quartets, **C3 alto clef**, AR 1.10–1.43) | 5 | **0.1475** | **0.0727** | **0.7236** | **3.6% (17 / 469, 3 Mozart works)** |
| **All 14 HW** | **14** | **0.2591** | **0.2174** | **0.7640** | **100% (469 rows)** |

*(Reference: `printed_clean` = `0.8968`, `printed_photo` = `0.8935` on the same checkpoint.)*

---

## 2. Audit of `real_hw` in `D:\tmp\uni_fullpage_iter11_clean.jsonl` (469 Rows)

### 2.1 Composer, Era, Scribe & Source Concentration

| Composer / Source | Era & Style | Rows | Share | Unique Works | Native Full Pages vs Strip-Composed | Clefs Present |
|---|---|---:|---:|---:|---|---|
| **Claude Debussy** (BnF Gallica) | Late Romantic / Impressionist (c.1890–1918), portrait French rag paper, neat steel/fountain pen | **452** | **96.4%** | **33** Gallica IDs | 79 native (`debussy-omr/train_pages`) + 373 strip-stitched (`real_hw`: 308, `real_hw_land`: 65) | `G2`, `F4` only (**0 C-clefs**) |
| **W. A. Mozart** (String Quartets) | Classical (1782–1790), landscape 4-staff quill autograph | **17** | **3.6%** | **3** (`k465`: 5, `k428`: 6, `k590`: 6) | **0 native** (17 strip-stitched in `real_hw_land`) | `G2`, `F4`, `C3` (36), `C4` (8) |
| **Beethoven, Schubert, Hensel, Mendelssohn, Cornelius, Bach, Chopin, Orchestral, Lieder** | 18th/19th-c. German/Austrian/Polish quill & iron-gall manuscripts | **0** | **0.0%** | **0** | 0 | — |

**Key Structural Deficits in `real_hw`:**
- **83.2% (`390 / 469`) are synthetically stitched strips**, not native full pages (`79 / 469` native, all Debussy).
- **Orientation:** `362` portrait (`77.2%`) vs `107` landscape (`22.8%`).
- **C-clef starvation:** Only **`17 / 469` (`3.6%`)** pages contain any C-clef (`clef-C-3`: 36, `clef-C-4`: 8 vs `clef-G-2`: 3,145 and `clef-F-4`: 2,037).
- **Systems/page:** Heavily skewed to 2–3 systems (`2s`: 226, `3s`: 132, `4s`: 71, `5+s`: 38); dense 5–7 system pages are scarce (`38 / 469 = 8.1%`).

### 2.2 Per-Piece Failure Map on `suite31_hw` (All 14 Historical HW Pieces)

| Piece | Stratum | Dims (AR) | Parts | Lyrics | Clefs | Sys | Ref Notes | Clean Note F1 | Photo Note F1 | Clean Tok F1 | HR Ratio | Clean Fixes |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `hw-hensel-morgengruss-d-b` | `hw_portrait_pdm` | `1586×2048` (0.77) | 4 | **276** | F4, G2 | 2 | 149 | **0.0245** | **0.0246** | 0.6517 | 0.64 | 239 |
| `hw-mendelssohn-herbstlied-d-b` | `hw_portrait_pdm` | `1613×2048` (0.79) | 3 | **105** | F4, G2 | 5 | 360 | **0.0704** | **0.0674** | 0.6936 | 1.76 | 924 |
| `hw-beethoven-op135` | `hw_landscape` | `2500×1934` (1.29) | 4 | 1 | **C3**, F4, G2 | 2 | 108 | **0.0797** | **0.0557** | 0.6949 | 1.56 | 254 |
| `hw-mozart-k387` | `hw_landscape` | `1191×853` (1.40) | 4 | 1 | **C3**, F4, G2 | 2 | 432 | **0.0834** | **0.0660** | 0.7399 | 1.44 | 967 |
| `hw-hensel-im-herbste-d-b` | `hw_portrait_pdm` | `1483×2048` (0.72) | 2 | **59** | F4, G2 | 3 | 354 | **0.0989** | **0.1085** | 0.6117 | 0.49 | 474 |
| `hw-mozart-k421` | `hw_landscape` | `1191×835` (1.43) | 4 | 1 | **C3**, F4, G2 | 2 | 818 | **0.1521** | **0.0744** | 0.6684 | 0.58 | 1,093 |
| `hw-schubert-d810` | `hw_landscape` | `2463×2244` (1.10) | 4 | 1 | **C3**, F4, G2 | 3 | 380 | **0.1810** | **0.1078** | 0.6733 | 0.72 | 534 |
| `hw-cornelius-botschaft-d-b` | `hw_portrait_pdm` | `1239×2048` (0.60) | 2 | **32** | F4, G2 | 4 | 235 | **0.2396** | **0.3409** | 0.8229 | 0.85 | 330 |
| `hw-mozart-k464` | `hw_landscape` | `1191×840` (1.42) | 4 | 1 | **C3**, F4, G2 | 2 | 415 | **0.2416** | **0.0594** | 0.8414 | 0.72 | 540 |
| `hw-debussy-lindaraja` | `hw_portrait_debussy` | `3358×4305` (0.78) | 2 | 0 | F4, G2 | 7 | 749 | **0.3186** | **0.2864** | 0.7982 | 1.11 | 1,078 |
| `hw-debussy-vetement-du-blesse` | `hw_portrait_debussy` | `3674×4658` (0.79) | 1 | 0 | F4, G2 | 6 | 278 | **0.3480** | **0.2081** | 0.7720 | 0.63 | 296 |
| `hw-debussy-la-plus-que-lente` | `hw_portrait_debussy` | `3696×5064` (0.73) | 1 | 0 | F4, G2 | 6 | 459 | **0.4306** | **0.5678** | 0.9473 | 0.82 | 476 |
| `hw-debussy-berceuse-heroique` | `hw_portrait_debussy` | `4354×6079` (0.72) | 2 | 0 | F4, G2 | 5 | 429 | **0.5652** | **0.4505** | 0.9265 | 1.04 | 380 |
| `hw-beethoven-das-blumchen-wunderhold-d-b` | `hw_portrait_pdm` | `1398×2048` (0.68) | 2 | **224** | F4, G2 | 4 | 232 | **0.7933** | **0.6259** | 0.8540 | 0.81 | 87 |

**Three crystal-clear failure clusters:**
1. **Landscape 4-staff C3 String Quartets** (`hw-beethoven-op135`, `hw-mozart-k387/k421/k464`, `hw-schubert-d810`): photo Note F1 collapses to **`0.056–0.108`** (low-res `1191×840` Mozart scans + cramped 4-staff landscape spacing + C3 alto clef + hurried quill stems/beams crossing staves).
2. **Dense German Kurrent/Sütterlin Cursive Lieder** (`hw-hensel-morgengruss` 276 lyrics → `0.0245`, `hw-mendelssohn-herbstlied` 105 lyrics → `0.0704`, `hw-hensel-im-herbste` 59 lyrics → `0.0989`): dense cursive lyric lines and editorial cross-outs between staves trigger massive hallucinations (`HR = 1.76` on Mendelssohn) or truncation (`HR = 0.49` on Hensel).
3. **Idle Real HW Corpora Already on `D:\tmp` (0% Used in `iter11`)**:
   - `D:\tmp\pdm-bach-chopin-hw`: **376 pages** (48 PDM works: 30 Bach chorales + 18 Chopin; 244 portrait + 132 landscape; SBB Berlin / SLUB Dresden).
   - `D:\tmp\pdm-openscore-hw`: **271 pages** (53 PDM Lieder works; 87 portrait + 184 landscape; CC0 OpenScore Lieder GT; exclude `suite31_hw` holdout slugs).
   - `D:\tmp\pdm-openscore-sq-hw`: **96 pages** (8 PDM String Quartet works, 4 autographs; 60 portrait + 36 landscape; CC0 OpenScore SQ GT; exclude eval overlap).
   - `D:\tmp\nifc-polish-hw`: **985 pages** (150 Polish historical manuscripts with `.krn`/MusicXML/bekern).
   - **Total: ~1,728 permissive real archival pages** with work-level symbolic GT sitting completely unused on disk.

---

## 3. Audit of Synthetic HW (`synth_hw`: 7,775 Rows) & `hw-score-sim`

### 3.1 Era, Implement & Mark Distribution Across `v04`, `v05`, and `v06`

In `uni_fullpage_iter11_clean.jsonl`, `domain="hw"` (`7,775` rows) comes strictly from `D:\tmp\hw-sim-v04-clean\pages` (**6,700 rows = 86.2%**) and `D:\tmp\hw-sim-v05-dense\pages` (**1,075 rows = 13.8%**). `D:\tmp\hw-sim-v06-historical-clean` (500 pages on disk) is **not yet in `iter11`**.

Using `hw_score_sim.catalog.describe_style()` across all 234 registered styles (`112` historical Renaissance–Impressionist vs **`122` = 52.1% modern 20th/21st-century**):

| Dataset | Rows in `iter11` | Historical Era (Ren–Impr) | **Modern Era (20c/21c)** | Unconstrained `random_scribe` | Landscape % | Pages w/ Cancellations | Pages w/ Verso Bleed Mark | Top Dominant `manuscript_marks` / Personas |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `hw-sim-v04-clean` (500/8,000 sample) | **6,700** (86.2%) | **34.0%** | **39.0%** | 27.0% | 46.0% | 9.2% | 63.4% | `chord_symbol` (567), `jazz_realbook_chart` (427), `film_studio_copyist` (239), `studio_timecode` (147) |
| `hw-sim-v05-dense` (555/1,664 sample) | **1,075** (13.8%) | **35.1%** | **40.7%** | 24.1% | 50.5% | 11.2% | 70.6% | `chord_symbol` (970), `jazz_realbook_chart` (622), `film_studio_copyist` (500), `studio_timecode` (153) |
| `hw-sim-v06-historical-clean` (500/500) | **0** (not in `iter11`) | **37.8%** | **37.2%** | 25.0% | 50.0% | **0.0%** | **0.0%** | `composer_working_draft` (1,260), `measure_number` (1,653), `library_stamp` (68) |

**Two major surprises exposed by the audit:**
1. **~40% of `synth_hw` (plus most unconstrained `random_scribe` draws) is wasted on modern 20th/21st-century styles** (`jazz_realbook_chart`, `film_studio_copyist`, `studio_timecode`, `chord_symbol`, ballpoint/gel/felt-marker pens, `spiral_notebook`, `ozalid_blue`, `soviet_green_tint`) that share almost zero ink/paper/layout physics with 18th–19th-century European manuscripts! Even `hw-sim-v06-historical-clean` forgot to filter `catalog.classify_era()` — **37.2% of `v06` still rendered modern/mid-century/contemporary styles** (including `gel_rollerball` and `ballpoint`).
2. **`v06` accidentally disabled cancellations (0.0%) and verso bleed-through marks (0.0%)**, removing two defining features of real historical autographs.

### 3.2 `hw-score-sim` Physics & Feature Gap Matrix vs Real 18th–20th-c. Manuscripts

| Visual / Structural Phenomenon | Currently in `hw-score-sim` (`v0.3.0`) | Gap vs `suite31_hw` & `rism_hw` |
|---|---|---|
| **Scribe styles & nibs** | 234 styles, 18 noteheads, 18 stems, 20 flags, 14 beams, 18 clefs, 18 accidentals, `nib_angle_deg`, `ink_pooling`, `ink_dropout` | Sampling is uniform over modern+historical (`52.1%` modern); `sample_random_scribe` is unconstrained across modern pens/papers. |
| **Staff droop, curvature & rastrum wear** | `staff_curve_px` (p50=`3.8 px`, p90=`8.8 px`, max=`15.8 px`), `staff_slope_deg` (p50=`0.5°`, max=`2.4°`) | No independent per-staff nonlinear S-curve/droop, no broken/faded inner staff lines, no uneven hand-ruled rastrum line spacing (`sp` jitter is uniform). |
| **Tight 4-staff landscape quartet crowding** | `intra_gap` hard-clamped `>= 3.6 * sp`; `synth_hw` has only `17.9%` C-clef pages and `13.9%` 4-staff systems in `v04`. | Real Mozart/Beethoven/Schubert landscape folios have cramped `2.2–3.0 * sp` inter-staff gaps where stems, beams, slurs, and ledger lines physically cross into adjacent staves. |
| **Dense cursive lyrics & title clutter** | Basic stick-stroke Latin text (`_draw_handwritten_text`), wide `7.2 * sp` lyric gap, simple 1-line headers. | Missing dense German Kurrent/Sütterlin & French cursive multi-verse lyrics (`59–276` syllables/page) that collide with stems/slurs/barlines, plus ornate archival title blocks & shelfmark stamps. |
| **Cross-outs / cancellations** | 7 `CANCELLATION_STYLES` in `manuscript_marks.py`, restricted by `SafeZoneGuard` to blank staves/margins (`9–11%` in `v04/v05`, **`0%` in `v06`**). | Missing **in-bar / adjacent-measure composer cancellations** (scribbled-out false starts on active staves that the OMR must skip without losing bar alignment). |
| **Verso ink bleed-through & iron-gall acid** | Capped at `bleed_through <= 0.25` → effective alpha `<= 0.07` (`min(bleed_through, 0.25) * 0.28`), and `0%` verso marks in `v06`. | Real RISM / SBB Berlin thin rag paper exhibits `0.15–0.35` alpha mirrored verso staves/noteheads + warm iron-gall haloing + dark archival border/gutter shadows. |

---

## 4. Audit of Unlabeled RISM, `CUT-RISM` (Exp 1), and Promoted `ov-39` Pseudo-Labels

### 4.1 Unlabeled RISM (`D:\tmp\unlabeled-10k-real\rism_hw`: 5,707 Pages)
- **Archive / Country Distribution:** `rism-de` (Germany: **5,144 = 90.1%**), `rism-ch` (Switzerland: **497 = 8.7%**), `rism-pl` (Poland: **66 = 1.2%**).
- **Orientation & Resolution (1/20 sample, `n=286`):** **45.1% landscape (`129`) / 54.9% portrait (`157`)**, 100% RGB, median `1441×1799`, AR `p10/p50/p90 = 0.61 / 0.80 / 1.37`. This is an **ideal visual match** for `suite31_hw`'s landscape quartets + portrait Lieder.
- **Complementary Archive:** `D:\tmp\unlabeled-10k-real\nifc_hw` (Polish manuscripts) can diversify the 90.1% German RISM skew.

### 4.2 `CUT-RISM` (Exp 1: `2026-11-26T00-54-cut-rism-real-ink-v06`) — Incomplete & Over-Constrained
Inspecting HF run ledger `runs/2026-11-26T00-54-cut-rism-real-ink-v06.json`:
- **Status:** Timed out on Modal L4 at **step 2,550 / 12,000** (`6,451 s`). **`datasets/cut_rism_v06_500.tar` was NEVER produced or uploaded.**
- **Diagnostic Failure Mode:** At step 2,550: `drift_px = 0.0014 px`, `ink_l1 = 0.0035`, `sobel_y_corr = 0.9953`, while **`loss_d = 0.0179`** and **`loss_g_gan = 0.9129`**. `geom_lock_weight = 4.0` locked ink pixels to near-identity, preventing quill/iron-gall texture transfer and letting the PatchGAN discriminator saturate at `loss_d ≈ 0.018`.

### 4.3 Promoted `ov-39` RISM Pseudo-Labels (`build_rism_pseudolabel_manifest.py`) — Ready But Unexecuted
- **Proven Lift (`ov-39` OUTCOME):** `mean_logprob >= -0.35` & `beat_valid_frac >= 0.70` (`26.0%` yield → ~1,484 pages / 5,707) achieved **`+0.0381` `hw_photo` F1** (`+26%` rel) and **`+0.0236` HW Avg F1** with zero printed drift (`-0.0006` clean / `+0.0033` photo). Tagging `beat_valid_frac >= 0.85` (`18.0%` yield → ~1,027 pages) at `1.75×` weight combines the `+0.0279` `hw_clean` and `+0.0714` `r_noloop` gains.
- **Current Gap:** `D:\tmp\rism-pseudolabels-ov39` **does not exist yet** — full 5,707-page pseudo-labeling has not been run, and `build_rism_pseudolabel_manifest.py` currently has no orientation/clef stratification (risking over-selection of easy 1–2 staff treble pages) and does not leverage the `~1,728` work-level PDM/NIFC pages on disk.

---

## 5. Concrete Iter 13 Data Diversity Upgrades (Ranked by Expected `suite31_hw` Impact)

1. **Generate `hw-sim-v07-historical-v2` (8,000–10,000 pages) with Strict Historical Filtering & Realistic Clutter (`hw-score-sim`):**
   - **Era/Implement Lock:** Sample **85% historical styles** (`era in {renaissance, baroque, classical, romantic, impressionist}` & `implement in {quill, steel_nib, fountain_pen, graphite_pencil}`; constrain `sample_random_scribe` to historical ranges) and **<=15% modern** (down from 40%+); disable `jazz_realbook_chart`, `film_studio_copyist`, `chord_symbol`, `studio_timecode` on historical pages.
   - **Targeted Layout & Clef Quotas:** **50% landscape** (`AR 1.15–1.50`), **>=35% 4-staff C3/C4 quartets** with tight `2.2–3.2 * sp` `intra_gap` (allowing realistic stem/slur inter-staff overlap), and **>=30% dense Lieder** with cursive-style multi-verse lyrics (`50–250` syllables/page) colliding near stems/slurs.
   - **Restore & Upgrade Historical Physics:** Re-enable **cancellations on 20–25% of pages** (including **in-bar crossed-out false-start measures** skipped by GT), raise **verso bleed-through alpha cap from `0.07` → `0.28`** (on 50% of historical pages), add per-staff nonlinear S-droop (`4–18 px`) + broken inner staff lines, and include low-res `1100–1300 px` Mozart-like scan downscale augmentation.
2. **Execute Promoted `ov-39` Full-Corpus RISM + NIFC Pseudo-Labeling (`~1,800–2,200` Real Archival Pages):**
   - Run `tmp-train/mvd/build_rism_pseudolabel_manifest.py` across all **5,707 `rism_hw` + `nifc_hw`** pages using the strongest Iter-12 / v2 teacher at `gen_max_len=2048`, `max_dim=1280`, with `ov-39` dual-tier gating (`lp >= -0.35`, `beat >= 0.70`, `1.75×` weight on `beat >= 0.85`).
   - Add **orientation & clef quota balancing** (cap portrait 1–2 staff treble pages so **>=45% of kept pseudo-labels are landscape** and C-clef / 3–4 staff systems are upweighted `2×`).
3. **Unlock the `~1,728` Idle PDM / NIFC Real Archival Pages on `D:\tmp` via Symbolic-GT-Verified Alignment:**
   - Run the teacher over `pdm-bach-chopin-hw` (376 pp), `pdm-openscore-hw` (271 pp, excluding `suite31_hw` holdouts), `pdm-openscore-sq-hw` (96 pp, excluding eval overlap), and `nifc-polish-hw` (985 pp), and **verify page predictions against each work's known MusicXML/bekern** (pitch/interval subsequence alignment). This converts weak work-level labels into **high-precision real Beethoven/Schubert/Hensel/Bach/Chopin/Quartet/Lieder full-page training data**, breaking the 96.4% Debussy monoculture.
4. **Fix & Complete `CUT-RISM` v2 (Exp 1 Iteration) to Translate `3,000+` Historical `hw-sim-v07` Pages:**
   - Lower `geom_lock_weight` from **`4.0` → `1.5`** (targeting `sobel_y_corr ∈ [0.88, 0.94]`, `drift_px <= 1.2 px` instead of `0.9953` identity lock), add **discriminator throttling / label smoothing (`0.9/0.1`)** whenever `loss_d < 0.08`, train on **era-filtered `hw-sim-v07` → `rism_hw`**, and translate `3,000` landscape quartet & Lieder pages into authentic 18th/19th-c. iron-gall ink & parchment.
5. **Rebalance `real_hw` Sampling & Retire Strip-Stitched Over-Weighting:**
   - Cap Debussy at **<=35%** of real HW draws (down from 96.4%) once PDM/NIFC + `ov-39` RISM pseudo-labels are added, and prioritize **native full pages** over strip-stitched synthetic pages (`83.2%` of current `real_hw`) to eliminate artificial inter-system seam artifacts.
