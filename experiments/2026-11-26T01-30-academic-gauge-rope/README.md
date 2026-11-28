# `2026-11-26T01-30-academic-gauge-rope` — Academic Experiment B: Gauge-RoPE (Diffeomorphism-Equivariant Intrinsic Rotary Embeddings)

- **Status:** `COMPLETED (ITERATE)` (`2026-11-26 02:15 UTC`, Modal `2× L4`, `ap-woqJUva6rjQCbQnyydluQ5`, full results in [OUTCOME.md](./OUTCOME.md))
- **Baseline Commit SHA:** `164852f`
- **Model:** `UniOMR v2 small_45m` (`~43M` params) + `GaugeFieldHead` (`138,587` params / `0.139M`, `< 0.15 MB` INT8)
- **Datasets (100% permissive):**
  - `gregoreesmaa/sheet-music-converter-omr` → `datasets/iter10_5_proxy_2k.tar` (`2,000` train + `64` val full pages)
  - `gregoreesmaa/sheet-music-converter-omr` → `datasets/rism_hw_5707.tar` (`5,707` unlabeled real historical RISM handwritten manuscripts for self-supervised diffeomorphic equivariance)
  - `gregoreesmaa/sheet-music-converter-omr` → `datasets/suite31_hw_eval.tar` (45-piece `suite31_hw` OOD benchmark: 31 printed + 14 real HW, `clean` + `photo` = 90 pages)

## Summary & Hypothesis
Curved, drooping, and hand-ruled staves in historical handwritten manuscripts break the Euclidean `(y, x)` grid assumption of standard `2D-RoPE`: a 3-staff-space vertical droop shifts a notehead's pixel `y` coordinate by half an octave across a system, causing pitch errors even when rhythm is read correctly.

**Gauge-RoPE** replaces fixed Euclidean rotary coordinates `(y, x)` with continuous learned intrinsic staff coordinates `(v̂, û) = (y + α·Δv̂, x + α·Δû)` predicted by a lightweight (`0.139M` param) anisotropic `GaugeFieldHead` with zero image resampling and a zero-initialized residual gate `α = 0.0` (exact Euclidean `RoPE2D` bit-identity at step 0). During training, `GaugeFieldHead` is additionally trained via self-supervised diffeomorphic equivariance (`diffeomorphic_equivariance_loss`) on unlabeled RISM manuscripts so it learns to rectify arbitrary smooth page warps/droops without any staff-line labels.

**Controlled 2-Arm Ablation (`1,000` steps each, identical `seed=42`, `lr=3e-4`, `dim=1280`, BF16 on Colab `L4`):**
- **Arm A (Control):** Standard Euclidean `RoPE2D(y, x)`.
- **Arm B (Treatment — Gauge-RoPE + RISM-SSL):** `attach_gauge_rope(model)` + `diffeomorphic_equivariance_loss` on unlabeled RISM batches (1 RISM SSL batch every 2 steps).

**Hypothesis:** Arm B will (1) exhibit substantially smaller degradation on `proxy_val` under synthetic 3-staff-space droop (`droop_frac=0.025`), (2) improve `suite31_hw` real handwritten (`hw_clean` / `hw_photo`) **Pitch-only token F1** and **Strict Note F1** relative to Arm A without regressing printed accuracy, and (3) learn a non-trivial residual gate `|α| > 0`.

## Quick Links
- Implementation: [gauge_rope.py](../../tmp-train/mvd/gauge_rope.py)
- CPU Unit Tests (6/6 PASS): [test_gauge_rope.py](../../tmp-train/mvd/test_gauge_rope.py)
- Colab Launcher: [run_colab.py](./run_colab.py)
- Plan & Decision Tree: [PLAN.md](./PLAN.md)
- Outcome: `OUTCOME.md` *(written immediately upon completion)*
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-26T01-30-academic-gauge-rope.json`
- HF Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-26T01-30-academic-gauge-rope.json`
