# `2026-11-27T14-50-i15-42-xf-aux-cbdice-unmix-gdcm`

- **Status:** `PLANNED` (P0 `wave=0` cross-field architectural proxy)
- **Short ID:** `i15-42`
- **Baseline Commit SHA:** `6055c49`

## Summary & Hypothesis
Tests **3 training-only cross-field auxiliary supervision heads (`aux_topo_unmix_gdcm`)** on `UniOMRv2Net(edge85m, ax0_gl6, d=512)` against the stock control (`ctrl`) under a controlled 2-arm 600-step BF16 `muon+adamw` proxy on `iter10_5_proxy_2k.tar` — with **exact `0.000 MB` inference weight delta** and **0% inference latency overhead** after `strip_aux_heads()`.

Combines three cross-field training-only regularizers:
1. **Paleography (Recto/Verso Ink-Layer Unmixing Head `InkUnmixHead`):** Early-stem `P2 (/4)` 2-channel head separating primary dark foreground ink from background paper & faint bleed-through, supervised on self-derived high-pass ink vs paper/bleed-through masks.
2. **Medical Vascular Imaging (`cbDice` Soft-Skeleton Thin-Line Preservation Loss):** Differentiable 3-iteration morphological soft-skeletonization (`clDice`/`cbDice`) applied to the existing `StaffSegHead` (`P2+P3`) auxiliary map so 1–2 px staff lines, stems, and ledger lines preserve topological connectivity.
3. **Handwritten Math Expression Recognition (`SemiHMER` Global Dynamic Counting Module `GDCM`):** Multi-query cross-attention counting head over pooled visual memory `mem_final` predicting per-page counts of 6 structural/symbol classes (`systems`, `barlines`, `notes`, `rests`, `accidentals`, `dots/ties`) via `SmoothL1Loss` in `log1p` space to curb premature `<eos>` and system/barline skips.

All experimental code lives strictly inside `run_colab.py` (`tmp-train/mvd/uni_omr_v2.py` untouched).

## Quick Links
- [PLAN.md](PLAN.md)
- [run_colab.py](run_colab.py)
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-27T14-50-i15-42-xf-aux-cbdice-unmix-gdcm.json`
- HF Partial Sync: `checkpoints/2026-11-27T14-50-i15-42-xf-aux-cbdice-unmix-gdcm/`
