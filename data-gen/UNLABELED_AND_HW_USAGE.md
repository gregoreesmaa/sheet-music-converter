# Unlabeled & Handwritten Corpora — Master Usage Guide (`data-gen/UNLABELED_AND_HW_USAGE.md`)

This guide documents every real handwritten and unlabeled sheet-music corpus staged on disk, its verified license, and **how to use each one cleanly without corrupting our `>= 0.9973` accuracy ceiling**.

> **Standing Quality Rule:** **Never** train Stage 2 (`MvdNet`) on raw single-model pseudo-labels from an inferior OMR model. All supervised transcriptions must be **human-authored** (CC0/CC-BY symbolic scores, RISM human-cataloged incipits, or verified 3-edition 100%-consensus prints).

---

## Corpus Inventory & Intended Use

| Corpus Path | Size | License | Labels | Best Use |
| :--- | :---: | :---: | :--- | :--- |
| **`D:\tmp\real-hw-corpus\`** *(Unified HW Registry: Route B + Quartets)* | **52 train works** (`48` OpenScore PDM 1.0 Lieder = `252` pages [`68` portrait, `184` landscape] + `4` Mozart SQ = `12` strips) + **5 commercial-safe portrait holdout works** in `suite31_hw` | **PDM 1.0** (SBB/SLUB) / **PD** × **CC0-1.0** (`OpenScore`) | `51/52` train + `5/5` eval post-`I1` `bekern` (`bekern-decode` rc=0) | **Primary supervised commercial-safe HW training** (`train_works.jsonl` via Sheet↔Symbolic DTW) + **zero-Gallica portrait holdout** (`suite31_hw_manifest.json` 45 pieces). |
| **`D:\tmp\rism-pdm-hw\`** *(Route A)* | **Scaling to 5,000 pages** (`52%` portrait / `48%` landscape) | **PDM 1.0** (SLUB/BSB/SBB) × **CC-BY 4.0** (RISM) | Human-cataloged **opening incipit** `bekern` (`~55–95` tok, 2–5 bars) | **1)** Stage-1 EMA geometric+scale equivariance; **2)** Crop opening staff via Stage 1 + barline/DTW align → **5,000 human-labeled HW opening-staff pairs** for Stage 2; **3)** Per-scribe style bank. |
| **`D:\tmp\real-hw-train\`** *(from `handwritten-500`)* | **4 works / 12 systems** (landscape string quartets) | **PD** (IMSLP) × **CC0** (`OpenScore SQ`) | Exact system-level post-`I1` `bekern` | Direct Stage-2 supervised HW training (`--real-hw-dir`). |
| **`D:\tmp\real-hw-train-debussy\`** *(33 works)* | **440 systems / 94 pages** (`89` portrait) | HF `CC-BY-4.0` *(BnF Gallica scan caveat: verify before commercial training)* | Human-transcribed MusicXML 4.0 → post-`I1` `bekern` | Research/benchmark & messy-autograph ablation; holdout 4 portrait works in `suite31_hw`. |
| **`D:\tmp\unlabeled-10k-real\`** | **9,977 full pages** (`p_music >= 0.65`) | **Public Domain** (IA/IMSLP/Wikimedia) | None (unlabeled) | **1)** Stage-1 EMA geometric + `log_hm` scale equivariance; **2)** Stage-2 encoder-only multi-view `mem`+`GDCM` consistency; **3)** Style bank; **4)** 3-print consensus mining. |
| **`D:\tmp\unlabeled-10k\hw_crops\`** | **629 HW music crops** (`p_music >= 0.65`) | **Public Domain** | None (unlabeled) | Stage-2 encoder-only multi-view consistency + combined per-scribe style reference pool. |

---

## Detailed Workflows

### A. Stage 1 (`StaffUNet2D`) — Unsupervised Equivariance on `unlabeled-10k-real` + `rism-pdm-hw`
1. Co-train `1:2` unlabeled:labeled pages.
2. Run slow EMA teacher (`τ=0.999`, `no_grad`) on clean unlabeled page `I`; run student on warped/degraded `Warp(I)`.
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
