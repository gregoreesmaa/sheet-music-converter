# Unlabeled & Handwritten Corpora — Master Usage Guide (`data-gen/UNLABELED_AND_HW_USAGE.md`)

This guide documents every real handwritten and unlabeled sheet-music corpus staged on disk, its verified license, and **how to use each one cleanly without corrupting our `>= 0.9973` accuracy ceiling**.

> **Standing Quality Rules:**
> 1. **Never** train Stage 2 (`MvdNet`) on raw single-model pseudo-labels from an inferior OMR model. All supervised transcriptions must be **human-authored** (CC0/CC-BY symbolic scores or verified 3-edition 100%-consensus prints).
> 2. **Never count RISM as labeled data** (RISM only has 2–5 bar monophonic opening incipits, not full-page/full-work transcriptions). All RISM manuscripts are grouped under **`D:\tmp\unlabeled-10k-real\rism_hw\`** as **unlabeled** data.
> 3. **CRITICAL `hw-sim-v03-clean` Warning:** Forensic audit proved `80.8%` of Stage-2 `hw-sim-v03-clean` strips had **barline corruption** and **`0%` `bekern+` expressive tokens**. Never ingest legacy `hw-sim-v03-clean` strips as-is — all synthetic HW strips must be exported **directly from source MusicXML** with a strict **`n_bar == m_end - m_start`** barline gate and restricted to **historical-only iron-gall / landscape / parchment styles**.
> 4. **Strict License Quarantine:** `cvc-muscima` is **`CC-BY-NC`** and is strictly quarantined from commercial training/data pipelines.

---

## Verified Real-HW & Unlabeled Inventory (On Disk)

- **Debussy archival full pages (uncomposed):** `84` pages in `D:\tmp\debussy-omr\train_pages\` *(research-only, BnF Gallica scan caveat)*
- **Debussy labeled system strips:** `440` strips (`D:\tmp\real-hw-train-debussy\`, research-only)
- **Mozart SQ labeled strips:** `36` strips (`D:\tmp\real-hw-corpus\`, commercial-safe)
- **`rism-pdm-hw-intl` (`PDM 1.0`):** `497` verified pages
- **Multi-page real HW works:** `258` works staged across our real-HW corpora
- **Unlabeled RISM / NIFC / intl HW pages:** `7,142` pages
- **Quarantined (`CC-BY-NC`):** `cvc-muscima` (excluded from commercial training)

---

## Corpus Inventory & Intended Use

### A. Commercial-Safe **LABELED** Real Handwritten Data (`D:\tmp\real-hw-corpus\`, full-work / system-level ONLY)
| Corpus Path | Size | License | Labels | Best Use |
| :--- | :---: | :---: | :--- | :--- |
| **`D:\tmp\real-hw-corpus\`** (`all_labeled_hw.jsonl`) | **52 train works / 288 images** (`48` OpenScore Lieder = `252` full pages [`68` portrait, `184` landscape] + `4` Mozart SQ = `36` strips) + **5 commercial-safe portrait holdout works** in `suite31_hw` | **PDM 1.0** (SBB/SLUB) / **PD** × **CC0-1.0** (`OpenScore`) | `51/52` train + `5/5` eval full post-`I1` `bekern` (`bekern-decode` rc=0) | **Primary supervised commercial-safe HW training** (via Sheet↔Symbolic DTW) + **zero-Gallica portrait holdout** (`suite31_hw_manifest.json` 45 pieces). |

### B. Research-Only Labeled HW (Kept Separate — BnF Gallica Caveat)
| Corpus Path | Size | License | Labels | Best Use |
| :--- | :---: | :---: | :--- | :--- |
| **`D:\tmp\real-hw-train-debussy\`** *(33 works)* | **440 systems / 94 pages** (`89` portrait) | HF `CC-BY-4.0` *(BnF Gallica scan caveat: exclude from commercial training)* | Human-transcribed MusicXML 4.0 → post-`I1` `bekern` | Research/benchmark & messy-autograph ablation; holdout 4 portrait works in `suite31_hw`. |

### C. **UNLABELED** Real Sheet-Music & Handwritten Corpora (`D:\tmp\unlabeled-10k-real\` + `unlabeled-10k\`)
| Corpus Path | Size | License | Labels | Best Use |
| :--- | :---: | :---: | :--- | :--- |
| **`D:\tmp\unlabeled-10k-real\images\`** | **9,977 full pages** (`p_music >= 0.65`) | **Public Domain** (IA/IMSLP/Wikimedia) | Unlabeled | **1)** Stage-1 EMA geometric + `log_hm` scale equivariance; **2)** Stage-2 encoder-only multi-view `mem`+`GDCM` consistency; **3)** Style bank; **4)** 3-print consensus mining. |
| **`D:\tmp\unlabeled-10k-real\rism_hw\`** *(DE + CH RISM, never counted as labeled)* | **5,300 full HW pages** (`5,000` DE `D-B/D-Dl/D-Mbs` + `300` CH `CH-Bu/CH-SO`; `2,863` portrait / `2,437` landscape) | **PDM 1.0** (SLUB/BSB/SBB/`e-manuscripta.ch`) | **Unlabeled** *(optional 2–5 bar opening incipits preserved in `rism-pdm-hw*`)* | **1)** Stage-1 EMA equivariance; **2)** **5,300-scribe commercial-safe style bank** for Structure-Locked ControlNet over `CC0 PDMX`; **3)** Stage-2 encoder-only EMA consistency. |
| **`D:\tmp\unlabeled-10k-real\intl_hw\`** *(Unlabeled CH)* | **450 verified HW music pages** (`126` MSS; **`76.7%` portrait** / `23.3%` landscape; CLIP mean `0.9950`) | **PDM 1.0** (`e-manuscripta.ch`: UB Basel `227`, ZB Zürich `217`, Solothurn `4`, Schaffhausen `2`) | Unlabeled | High-portrait Swiss handwritten pool for Stage-1 equivariance, Stage-2 encoder-only EMA consistency, and Per-Scribe ControlNet style bank. |
| **`D:\tmp\unlabeled-10k\hw_crops\`** | **629 HW music crops** (`p_music >= 0.65`) | **Public Domain** | Unlabeled | Stage-2 encoder-only multi-view consistency + combined per-scribe style reference pool. |

---

## Detailed Workflows

### A. Stage 1 (`StaffUNet2D`) — Unsupervised Equivariance on `unlabeled-10k-real` + `rism-pdm-hw`
> **Verified 2×2 Stage-1 Ablation Verdict:** Unsupervised EMA **hurt Printed Clean** (`-4.7%` to `-8.0%`) while clean-paper Hero HW **helped** (`+6.3%` to `0.7482`). Therefore, keep EMA **OFF** during supervised warmup and only gate it on after **real-HW F1 > `0.35`**.

1. During supervised warmup, keep EMA **OFF** and lean on clean-paper Hero HW + supervised labeled pages.
2. Once **real-HW F1 > `0.35`**, optionally enable EMA co-training (`1:2` unlabeled:labeled pages): run slow EMA teacher (`τ=0.999`, `no_grad`) on clean unlabeled page `I`; run student on warped/degraded `Warp(I)`.
3. Enforce `Student(Warp(I)) == Warp(Teacher(I))` on `ch0–2` (confidence `>0.85`, OOB masked) + exact scale law `log_hm(Scale_s(I)) == log_hm(I) + log(s)` on `ch3` + horizontal flip consistency on `ch2`.
4. Optionally add **Differentiable 5-Line Ribbon Comb-Energy** through `BezierRibbonSampler` on high-confidence unlabeled staves.

### B. Stage 2 (`MvdNet`) — Turning `rism-pdm-hw` & `pdm-openscore-hw` into Supervised HW Strips
1. **Route A (`rism-pdm-hw`, opening incipits):** Run `StaffUNet2D` on page 1 → extract top system/staff → match opening barlines to the incipit's measure count (verifying via DTW onset margin) → train Stage 2 on verified `(opening_staff, incipit_bekern)` pairs.
2. **Route B (`pdm-openscore-hw`, full pages):** Run `StaffUNet2D` across all pages of each work → align system ribbons to the full `CC0` MusicXML via **Sheet↔Symbolic monotonic DTW** (strict barline-count & margin gate) → exact human-labeled multi-system portrait & landscape handwritten strips!

### C. Stage 2 (`MvdNet`) — Unsupervised Consistency & Zero-Noise Style Synthesis
1. **Encoder-Only Multi-View EMA Consistency (zero token labels):** On unlabeled strips (`unlabeled-10k-real` + `hw_crops`), minimize masked cosine distance on `mem` + `SmoothL1(log1p(gdcm))` between weak EMA view `U_w` and photometric-only `U_s`.
2. **Structure-Locked ControlNet + Per-Scribe IP-Adapter:** Train Apache/MIT SD1.5-LCM/SDXL-Turbo LoRA+ControlNet strictly on our commercial-safe `PDM 1.0` scans (`rism-pdm-hw` + `pdm-openscore-hw` + `unlabeled-10k-real`). Repaint exact-labeled `CC0 PDMX` semantic masks in any unlabeled scribe's hand, gated by a **BBox Ink-Mass Gate** (`≥0.35` dark-pixel recall inside every GT accidental/dot/notehead box) for `0%` label corruption.

*(Mirrored local guides also live at `D:\tmp\real-hw-corpus\USAGE.md`, `D:\tmp\rism-pdm-hw\USAGE.md`, `D:\tmp\unlabeled-10k-real\USAGE.md`, and `D:\tmp\unlabeled-10k\USAGE.md`.)*

---

## Next International Commercial-Safe Archive Targets (Audited 2026-11-23)

1. **BNE Digital / Biblioteca Digital Hispánica (Spain):** Blanket **`CC-BY 4.0`** (explicit free commercial reuse) + full IIIF + bulk `BNElab` CSV/JSON (`Partituras manuscritas`, thousands of 18th–19th c. CWMN MSS).
2. **Alvin Portal (Sweden — Uppsala Düben Coll., Musikverket, Lund/Gothenburg):** **`PDM 1.0`** scans + **`CC0`** metadata via OAI-PMH (`alvin/OAI2`) + IIIF (`2,300+` Düben Baroque/Classical works + Musikverket/Leufsta).
3. **e-manuscripta.ch (Switzerland — ZB Zürich, UB Basel, ETH, Swiss NL):** **`PDM 1.0`** + full IIIF + tight RISM Switzerland (`CC-BY` PAE incipits) cross-linking.
4. *Runners-up:* **Det Kgl. Bibliotek (`kb.dk`, Denmark)** (`PDM` Carl Nielsen + CNW MEI), **Nasjonalbiblioteket (`nb.no`, Norway)** (`Falt i det fri` Grieg), **Moravian Library (`Kramerius`, Czech Republic)** (`Pouze veřejné`), **Meertens Tune Collections (Netherlands)** (`CC-BY 4.0` symbolic `**kern`).
5. *Excluded Legal Traps:* `MusiCorpus` (`CC-BY-NC-SA 4.0`), Finna.fi Sibelius (d. 1957 → copyrighted until 2028-01-01), Levande Musikarv PDFs, Italy `Internet Culturale` / Bologna, UK `Fitzwilliam`/`RCM`/`Bodleian` NC, Belgium `KBR`.
