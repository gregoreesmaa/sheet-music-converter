# Experiment `2026-11-26T00-54-cut-rism-real-ink-v06` — Geometry-Locked CUT-RISM Real-Ink Style Transfer

- **Status:** `RUNNING` (launched detached on Modal `L4`)
- **Baseline Commit SHA:** `9d73170`
- **Owner / Platform:** Modal `L4` (`24 GB`, Ada Lovelace `cc=8.9`, native BF16), profile `gregoreesmaa1`

## Summary & Hypothesis

Train our Geometry-Locked **CUT (Contrastive Unpaired Translation)** real-ink style transfer model (`tmp-train/mvd/train_cut_rism.py`, ~11.4M-param ResNet generator + PatchGAN discriminator + multi-layer PatchNCE + high-pass ink/Sobel-Y geometry lock) unpaired between our 500 synthetic `hw-sim-v06-historical-clean` pages (`datasets/hw_sim_v06_500.tar`) and our 5,707 real 18th–19th-century RISM archival manuscripts (`datasets/rism_hw_5707.tar`).

**Hypothesis:** Translating all 500 `hw_sim_v06` pages into authentic RISM iron-gall ink, quill-stroke variation, bleed-through, and aged parchment textures — while locking ink centroids and staff-line vertical edges (`<= 1.5 px` drift) so MusicXML / `bekern+` ground truth remains pixel-exact — will close the sim-to-real visual gap for handwritten OMR (`suite31_hw` `hw_clean` and `hw_photo` Strict Note F1) when mixed into downstream UniOMR training.

## Quick Links

- **Execution Plan & Live Query IDs:** [PLAN.md](./PLAN.md)
- **Self-Contained Modal Launcher:** [run_modal.py](./run_modal.py)
- **Outcome (written on completion):** `OUTCOME.md` *(pending)*
- **Hugging Face Repo (`gregoreesmaa/sheet-music-converter-omr`, private dataset):**
  - Run ledger: `runs/2026-11-26T00-54-cut-rism-real-ink-v06.json`
  - Rolling checkpoints & previews: `checkpoints/2026-11-26T00-54-cut-rism-real-ink-v06/`
  - Translated output dataset: `datasets/cut_rism_v06_500.tar`
