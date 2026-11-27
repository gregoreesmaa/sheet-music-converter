# Experiment: `2026-11-27T08-19-i15-09-data-mix-ratio-and-rep-cap-sweep`

- **Title:** Wave 13 Data-Mix Ratio & Archival Repetition-Cap Controlled Sweep (`arm0_v3_50_50` vs `arm1_hw_heavy_30_70` vs `arm2_rep_cap_10x_hard_upsample`, 800 Steps/Arm Warm-Start on Frozen `UniOMRv2` EMA)
- **Status:** `QUEUED` (Wave 13, Colab H100/L4, ~45 min L4 / ~10 min H100, ~3.5 CU)
- **Baseline Commit SHA:** `b12a003`
- **Baseline Checkpoint:** Frozen `UniOMRv2` EMA (`checkpoints/iter14_v2a_best/uni_omr_v2_ema_best.pt` → fallback chain down to `checkpoints/iter10_5_v2_best/uni_omr_v2_ema_best.pt` on HF `gregoreesmaa/sheet-music-converter-omr`)

## Summary & Hypotheses

In Iter 13/14 full-page training, `real_hw_archival` contains only a few hundred unique archival pages, so naive bucket upsampling (`12%–18%` of every step) can repeat individual archival pages `>20–40×` while starving diverse synthetic/pseudo-labeled HW and hard multi-staff layouts (landscape 4-staff, lead-sheet/lyrics, dense 5+ staff). Conversely, shifting the overall Printed:HW ratio from `50:50` (`v3`) to `30:70` may boost handwritten generalization or trigger printed regression.

This controlled 3-arm warm-start ablation isolates **data-mix ratio** and **per-unique-page repetition capping + hard-layout upsampling** under identical `800`-step budget, optimizer, LR schedule, seed (`42`), and full 90-page `suite31_hw` evaluation (both RAW and EMA):
- **Arm 0 (`arm0_v3_50_50` — Control):** Baseline Iter 14 `v3` mix (`50%` Printed / `19.5%` Synth-HW / `12%` Real-HW / `12%` RISM-Pseudo / `6%` Real-Stitched / `0.5%` Neg).
- **Arm 1 (`arm1_hw_heavy_30_70`):** HW-heavy mix (`30%` Printed / `25%` Synth-HW / `18%` Real-HW / `20%` RISM-Pseudo / `6.5%` Real-Stitched / `0.5%` Neg) testing whether higher HW exposure improves `suite31_hw` HW Note F1 without unacceptable printed drift.
- **Arm 2 (`arm2_rep_cap_10x_hard_upsample`):** `50/50` mix WITH a strict per-unique-page repetition cap (`<= 10×` on `real_hw_archival`, cleanly spilling excess draw weight to `rism_pseudo` + `synth_hw`) + `2×` upsampling of hard layouts (landscape 4-staff, lead-sheet/lyrics, dense 5+ staff).

**Promotion Gate:** Any candidate arm achieving **`>= +2.0 pp` HW Note F1** over `arm0_v3_50_50` with **`<= 0.5 pp` Printed Note F1 drop** on `suite31_hw` (all 90 pages, verified on both RAW and EMA) is promoted directly into the Iter-15 data sampler.

## Quick Links

- Plan & launch spec: [PLAN.md](PLAN.md)
- Runner script: [run_colab.py](run_colab.py)
- Queue ledger: [../overnight_queue/queue.json](../overnight_queue/queue.json) (`short_id: "i15-09"`, `wave: 13`, `priority: 9`)
- Outcome (written upon completion): `OUTCOME.md`
- HF Eval JSON: `evals/2026-11-27T08-19-i15-09-data-mix-ratio-and-rep-cap-sweep.json`
- HF Run Ledger: `runs/2026-11-27T08-19-i15-09-data-mix-ratio-and-rep-cap-sweep.json`
