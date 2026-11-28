# Co-Evolutionary Simulator ↔ Classifier Realism Strategy

**Status:** Active Grand Strategy (Iter 16+)  
**Simulator:** `hw-score-sim` + `photo-degrade` (`doc-photo-degrade`)  
**Benchmark Harness:** `hw_score_sim.realism_bench` (`python -m hw_score_sim.realism_bench`)  
**Reference Real Corpus:** `D:\tmp\unlabeled-10k-real\rism_hw` (5,707 historical manuscripts) + archival Debussy/NIFC/PDM pages

---

## 1. Core Principle: Never Freeze the Adversary

Optimizing a procedural handwriting simulator (`hw-score-sim` → `photo-degrade`) against a **fixed** discriminator inevitably hits **Goodhart's Law**: the simulator learns to fool that specific small CNN while stronger visual encoders (like UniOMR's own ConvNeXt + ViT backbone) still see a wide sim-to-real domain gap.

Therefore, closing the handwritten OMR sim-to-real gap is a **two-sided co-evolutionary loop**:

```text
  [1. Benchmark]  Run realism_bench (hw-score-sim -> photo-degrade vs real rism_hw)
        │           └─ Must pass Real-vs-Real Null Control (~0.50 AUC)
        ▼
  [2. Diagnose]   Read orthogonal 6-view AUCs, Probe C spectral/feature shifts,
        │         bilateral HiResCAM overlays (+fake / -real), & top_uncovered_real
        ▼
  [3. Upgrade     Fix top ranked subsystem_scorecard gaps (nib/stroke, scale/density,
      Simulator]  layout/margins, illumination/bleed, scribe/paper coverage)
        │           └─ Guard against destructive realism (Legibility SNR + OMR eval)
        ▼
  [4. Escalate    Whenever current adversary drops below ~0.75 AUC (or saturates),
      Classifier] upgrade realism_bench to the next Adversarial Tier -> repeat [1]
```

---

## 2. Adversarial Classifier Escalation Ladder

Every classifier tier MUST pass the **5-repeat Real-vs-Real Null Control** (`|mean_null_auc - 0.50| <= 0.12`, symmetric 95% CI covering `0.50`) and **manuscript-level group CV** before its gap attributions are trusted.

| Tier | Adversary Architecture | What It Catches | Escalation Trigger |
| :---: | :--- | :--- | :--- |
| **Tier 1 (Live)** | Dilated 4-block `PatchForensicCNN` (6 orthogonal views: `rgb`, `gray`, `gray_norm`, `lowfreq_gray`, `highfreq_gray`, `binary`) + CoordConv `FullPageForensicCNN` (`gray_norm`) + 55-dim spectral/Gabor/color Probe C + k-NN PRDC Coverage | Paper/ink color, exposure, mottling/bleed, nib gradient sharpness, binary stroke morphology, macro margins/density, mode collapse | Active baseline (`v0.7` → `v0.8`). Escalate once `patch_binary < 0.75` and `knn_coverage >= 0.50`. |
| **Tier 2 (Next)** | **Frozen OMR & Foundation Vision Backbones** plugged into Probe C `embed_fn`: **(a) Frozen UniOMR visual encoder** (ConvNeXt + ViT features) and **(b) Frozen `DINOv2` / `ConvNeXt`** | Domain shift inside the **exact representation space that drives OMR Strict Note F1**, plus high-level semantic glyph/layout realism invisible to hand-crafted Gabor/FFT features | Wire immediately after `v0.8` baseline so every simulator iteration directly tracks **UniOMR-space FD & Coverage**. |
| **Tier 3 (Hardest)** | **System-Strip (`512×128`) Multi-Bar Discriminator** + **Cross-Archive Held-Out CV** (train discriminator on Archive A e.g. `D-B`, test on Archive B e.g. `PL-Wn`/`CH-Bu`) | Multi-bar rhythmic spacing, clef/key/time-to-notehead cadence, beam slants, inter-staff lyric/chord clutter, and archive-invariant handwriting realism (prevents classifier from cheating on library-specific scanner beds) | Escalate once Tier 2 UniOMR-encoder linear probe AUC drops `< 0.70`. |

---

## 3. Non-Negotiable Guardrails (Preventing Shortcut & Collapse Traps)

1. **Mode-Collapse / Diversity Gate (Precision + Recall/Coverage):**
   - Never optimize AUC alone. A simulator that clones 1–2 neat real styles can drive AUC down while destroying OMR generalization.
   - Track **k-NN Precision, Recall, Density, Coverage, `variance_ratio`**, and **`top_uncovered_real` pages** on every run. `INDISTINGUISHABLE` requires **both** low AUC **and** `knn_coverage >= 0.70`.
2. **Anti-Shortcut Canonicalization:**
   - Identical optical pre-blur + JPEG Q75 round-trip before AND after staff-scale normalization (`sample_ink_patches`), `border_margin_frac = 0.10`, edge-gated ink sampling `[0.04, 0.35]`, `pad_mode="reflect"` + `gray_norm` on Probe B, and manuscript-level CV grouping (`infer_manuscript_groups`).
3. **Anti-Destructive-Realism Gate (Legibility SNR & OMR F1):**
   - Cranking up ink dropout, bleed, or blur can fool a discriminator while destroying ground-truth notation legibility.
   - Monitor `michelson_contrast_mean` and `staff_peak_prominence_mean` in `domain_stats` AND validate every major simulator release (`v0.8`, `v0.9`, ...) on **actual `suite31_hw` HW Strict Note F1** in Iter 16+.

---

## 4. Measured Progress Ledger (`hw-score-sim → photo-degrade` vs real `rism_hw`, 400 pages / 2,400 patches)

| Version | Tier | Null AUC | `patch_rgb` | `patch_highfreq` | `patch_binary` | `fullpage` | `stdFD` | `Coverage` (`null`) | `sp_px` Real / Sim | `Ink%` Real / Sim |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`v0.7` baseline** (`5e7e4fe`) | T1 | `0.535` (PASS) | `0.9940` | `0.9869` | `0.9722` | `0.9987` | `69.69` (`5.63`) | `0.165` (`0.966`) | `7.0` / `11.5` (`1.64×`) | `19.3%` / `10.9%` |
| **`v0.8`** *(running)* | T1 | *pending* | *pending* | *pending* | *pending* | *pending* | *pending* | *pending* | *pending* | *pending* |

## v0.8 benchmark result (measured 2026-11-28, same protocol as v0.7; report D:\tmp\hwsim-realism-bench-v08\report.json)

| Metric | v0.7 | v0.8 |
|---|---:|---:|
| Probe C AUC | 0.9967 | 0.9922 |
| Probe C stdFD | 69.7 | 62.2 |
| recall / coverage | 0.185 / 0.165 | 0.260 / 0.180 |
| variance_ratio | 0.354 | 0.432 |
| patch_binary AUC | 0.972 | 0.981 (worse) |
| patch_highfreq AUC | 0.987 | 0.998 (worse) |
| fullpage AUC | 0.9987 | 0.9993 |
| staff spacing p50 (real 7.0) | 11.5 | 12.0 (worse) |
| ink fraction (real 19.3%) | 10.9% | 13.3% |
| patch fallback (real 4.1%) | 46.3% | 36.0% |

Remaining gaps: staff-scale mismatch 1.71x (pack 8-14 staves per page), high-frequency/edge energy too low vs real, coverage/diversity still low, full-page framing. PARKED until agent credits return.
