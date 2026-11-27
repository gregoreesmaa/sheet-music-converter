# `2026-11-27T14-55-i15-50-xf-remora-modifiers-daq-fape`

- **Status:** `PLANNED` (QUEUED in `experiments/overnight_queue/queue.json`, Wave 0 / P0)
- **Short ID:** `i15-50`
- **Baseline Commit SHA:** `6055c49`
- **Cross-Field Origin:** Nanopore Genomics (`Remora` Move-Table decoupled modifier head) + Cryo-EM (`ModelAngelo` 2D-FAPE local-frame offset loss + `DAQ` independent local raster-vs-symbol register-shift critic).

## Summary & Hypothesis
Controlled 2-arm weight-incompatible architectural proxy on `UniOMRv2Net(edge85m, ax0_gl6, d=512)` using `iter10_5_proxy_2k.tar` (600-step BF16, verified `muon+adamw`):
- **Arm 0 (`ctrl`):** Standard `edge85m + ax0_gl6`.
- **Arm 1 (`remora_daq_fape`):** Adds three tightly-scoped cross-field modules:
  1. **`RemoraModifierHead` (`~0.3 MB` INT4):** Lightweight local cross-attention refinement head (`d_mod=128`) attending from decoder states to encoder memory, emitting a gated residual logit delta strictly masked to rare modifier/accidental/dot/articulation tokens (`a-2..a2`, `dot`, `tie-*`, `stacc`, `accent`, `tenuto`, `fermata`, `slur-*`, `art-*`, `grace`, `cue`, `dyn-*`, `hairpin-*`).
  2. **Training-only `FAPE2DOffsetHead` (`0.00 MB` inference, stripped at export):** Predicts local-frame relative diatonic pitch-step & octave offsets `(Δstep, Δoct)` between consecutive notes within a bar under clamped L1 loss (translation-invariant to global clef/staff-y shift, inspired by Cryo-EM `ModelAngelo` 2D-FAPE).
  3. **`DAQRegisterCriticHead` (`~0.1 MB` INT4):** Independent local visual-vs-token consistency critic trained via BCE on positive real bars vs synthetic octave/step-shifted negative bars to flag silent register shifts at inference.

**Hypothesis:** Decoupling rare tiny modifiers into a dedicated `Remora` head prevents frequent notehead gradients from drowning out accidentals/dots/articulations, training-only 2D-FAPE enforces local melodic interval geometry at zero inference cost, and `DAQ` provides an independent visual-symbol register-shift detector at `< 0.5 MB` total INT4 overhead (`packed_int4_mb < 55.0 MB`).

## Quick Links
- [PLAN.md](PLAN.md)
- [run_colab.py](run_colab.py)
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-27T14-55-i15-50-xf-remora-modifiers-daq-fape.json`
- HF Partial Sync: `checkpoints/2026-11-27T14-55-i15-50-xf-remora-modifiers-daq-fape/`
