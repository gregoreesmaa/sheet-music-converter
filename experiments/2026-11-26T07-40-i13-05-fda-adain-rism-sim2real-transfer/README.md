# `2026-11-26T07-40-i13-05-fda-adain-rism-sim2real-transfer`

- **Experiment ID:** `2026-11-26T07-40-i13-05-fda-adain-rism-sim2real-transfer`
- **Short Code:** `i13-05` (Iter 13 Academic Scout 2 — Unpaired Sim-to-Real Ink/Paper Domain Transfer)
- **Status:** `QUEUED`
- **Predecessor / Lineage:** `ov-39` (`2026-11-26T01-38-ov-39-rism-pseudolabel-threshold-sweep`, `+0.0381` `hw_photo` lift from exposing `uni_omr_soup.pt` to real RISM pages)
- **Academic Basis (2024–2026):**
  - **Fourier Domain Adaptation (FDA) & Unpaired Document Style Transfer** (Yang & Soatto; 2024–2026 Historical Document Analysis & OMR sim-to-real literature alongside CUT / CycleGAN-Turbo): swapping only the low-frequency 2D FFT amplitude spectrum ($\beta \in \{0.02, 0.06\}$) from unlabeled real historical manuscripts (`rism_hw_5707`) onto labeled synthetic/printed scores transfers authentic parchment illumination, sepia foxing, vignetting, and bleed-through envelopes while preserving 100% of the high-frequency phase spectrum (exact 1–2 px staff lines, ledger lines, noteheads, and exact `bekern` GT alignment — zero symbol hallucination).
  - **Chromatic-AdaIN / MixStyle Moment Transfer** (2024–2026 HDA/HTR domain generalization): aligns per-channel ink/parchment mean and variance statistics to real archival RISM pages online during training.

## Hypothesis (Single Variable)

Applying online unpaired **RISM-5707 Domain Transfer** (2D FFT low-frequency amplitude swap `FDA` or `Color-AdaIN` moment matching with `p=0.5`) to **ground-truth labeled** `iter10_5_proxy_2k` training pages during `120`-step fine-tuning of `uni_omr_soup.pt` will close the archival sim-to-real gap on `suite31_hw` (`hw_clean` / `hw_photo`) by `>= +0.015` Note F1 without pseudo-label noise and without regressing printed scores.

## Arms (4 Matched Arms, Identical Seed & Batch Order)

1. `arm_ctrl_standard`: Standard mild contrast/brightness jitter (control).
2. `arm_fda_b002`: Unpaired RISM-5707 FDA low-frequency amplitude swap (`beta=0.02`, `p=0.5`).
3. `arm_fda_b006`: Unpaired RISM-5707 FDA low-frequency amplitude swap (`beta=0.06`, `p=0.5`).
4. `arm_color_adain`: Unpaired RISM-5707 Chromatic-AdaIN moment transfer (`alpha ∈ [0.4, 0.8]`, `p=0.5`).
