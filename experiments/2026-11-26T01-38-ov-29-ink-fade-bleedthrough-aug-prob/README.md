# `2026-11-26T01-38-ov-29-ink-fade-bleedthrough-aug-prob` — Yellowed-Paper, Ink-Fade & Mirrored Bleed-Through `aug_prob` Sweep

- **Status:** `QUEUED` (Overnight Wave 6, ready for Google Colab `L4`)
- **Baseline Commit SHA:** `164852f`
- **Model:** `UniOMR v2 small_45m` (`~43M` params, `d_model=384`, 2 axial + 4 global + 8 GQA dec, MTP-4)
- **Datasets (100% permissive):**
  - `gregoreesmaa/sheet-music-converter-omr` → `datasets/iter10_5_proxy_2k.tar` (`2,000` train + `64` val full pages, 0 `suite31_hw` overlap)
  - `gregoreesmaa/sheet-music-converter-omr` → `datasets/suite31_hw_eval.tar` (45-piece `suite31_hw` OOD benchmark: 31 printed + 14 real HW, `clean` + `photo` = 90 pages)

## Summary & Hypothesis
Historical manuscript scans (RISM, Debussy, Mozart, Chopin) and aged printed scores suffer heavily from yellowed parchment casts, spatially uneven iron-gall ink fade, and ghosted recto/verso bleed-through from the reverse side of thin paper. While `uni_omr.py` defaults to `aug_prob=0.75`, setting the archival degradation probability too high can starve the model of crisp engraved contrast, whereas setting it too low leaves `hw_clean`/`hw_photo` vulnerable to low-contrast faint strokes and verso ghosting.

This 3-arm controlled sweep isolates **`aug_prob ∈ {0.30, 0.55, 0.75}`** using a dedicated **yellowed-paper + spatially varying ink-fade + horizontally-mirrored blurred verso bleed-through** photometric degradation pipeline on `iter10_5_proxy_2k.tar` (`UniOMR v2 small_45m`, `600` steps per arm, identical `seed=42`, `lr=3e-4`, `dim=1280`, BF16 on Colab `L4`).

**Hypothesis:** `aug_prob=0.55` will achieve the best Pareto trade-off between `suite31_hw` `printed_clean` precision and `hw_clean` / `hw_photo` + archival-stress-test robustness, outperforming `0.30` on faded/bleed-through pages and outperforming `0.75` on clean engraved scores.

## Quick Links
- Colab Launcher: [run_colab.py](./run_colab.py)
- Plan & Decision Tree: [PLAN.md](./PLAN.md)
- Queue Manifest: [batch_06.json](../overnight_queue/batch_06.json)
- Outcome: `OUTCOME.md` *(written immediately upon completion)*
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-26T01-38-ov-29-ink-fade-bleedthrough-aug-prob.json`
- HF Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-26T01-38-ov-29-ink-fade-bleedthrough-aug-prob.json`
