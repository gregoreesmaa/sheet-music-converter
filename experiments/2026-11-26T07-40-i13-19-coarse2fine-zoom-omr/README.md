# `2026-11-26T07-40-i13-19-coarse2fine-zoom-omr` — Coarse-to-Fine 2-Pass System-Plan → Native-Res Zoomed-System Foveated Decoder

- **Experiment ID:** `2026-11-26T07-40-i13-19-coarse2fine-zoom-omr`
- **Proposal Slot:** `i13-19` (Iter 13 Risky Bets — Batch 07, Moonshot 1)
- **Status:** `PROPOSED`
- **Platform / GPU:** Google Colab `1× NVIDIA L4` (`24 GB` VRAM, BF16 native, ~42 min, ~3.5 CU)
- **Model Footprint:** `UniOMR v2 small_45m` (`~41.3M`) + `SystemBandProposalHead` & `GlobalContextCarryProj` (`~0.24M`) = **`~41.5M` params** (`~22.2 MB` INT4 / `~30.5 MB` Hybrid FP16/INT4, well inside `< 55 MB` WebGPU budget)
- **Datasets (100% permissive):**
  - `gregoreesmaa/sheet-music-converter-omr` → `datasets/iter10_5_proxy_2k.tar` (`2,000` train + `64` val full pages)
  - `gregoreesmaa/sheet-music-converter-omr` → `datasets/suite31_hw_eval.tar` (45-piece `suite31_hw` OOD benchmark: 31 printed + 14 real historical HW, `clean` + `photo` = 90 pages)

## Moonshot Thesis (Why This Could Jump `hw_clean` `0.26 -> 0.60+`)
At `1280px` full-page resolution, a dense 6–8 system historical manuscript allocates only ~140–180 vertical pixels per system; after `UniOMR v2`'s `/16` `PatchMerger2x2`, an entire 5-line staff collapses into ~4–6 vertical tokens. Tiny handwritten accidentals, ledger lines, and closely spaced chord heads blur together, while the single-pass autoregressive decoder loses its vertical `<sys>` pointer across drooping or unevenly spaced staves.

`Coarse2FineZoomOMR` reuses the **exact same `41.3M` weights** in a **2-pass foveated loop** with only `+0.24M` parameters:
1. **Pass 1 (Coarse Global Plan, `768px`):** Encodes the full page at low resolution (~6 ms) and predicts vertical system bands `[y_top, y_bot]` via `SystemBandProposalHead` plus a 4-token **Global Page Context** summary (key/clef/instrument prior).
2. **Pass 2 (Foveated Native-Res Zoom, `256×1280` per band):** Crops each predicted band with `±12%` overlap margin directly from the native high-res image (**3–5× higher effective vertical DPI per staff space**), encodes with the shared `UniOMR v2` encoder, prepends the Global Context tokens, and decodes each system cleanly without `<sys>` pointer drift.

## Primary Risk
If Pass 1 merges or misses curved/bracketed grand-staff systems on messy archival manuscripts, Pass 2 will drop or duplicate entire systems.

## Quick Links
- Detailed Plan & Kill/Promote Gates: [PLAN.md](./PLAN.md)
- Standalone Colab L4 Runner: [run_colab.py](./run_colab.py)
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-26T07-40-i13-19-coarse2fine-zoom-omr.json`
- HF Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-26T07-40-i13-19-coarse2fine-zoom-omr.json`
