# `2026-11-26T07-40-i13-20-masked-diffusion-measure-denoiser` — Bidirectional Masked-Diffusion Measure Denoiser (`MDM-MeasureRefiner`)

- **Experiment ID:** `2026-11-26T07-40-i13-20-masked-diffusion-measure-denoiser`
- **Proposal Slot:** `i13-20` (Iter 13 Risky Bets — Batch 07, Moonshot 2)
- **Status:** `PROPOSED`
- **Platform / GPU:** Google Colab `1× NVIDIA L4` (`24 GB` VRAM, BF16 native, ~40 min, ~3.4 CU)
- **Model Footprint:** `UniOMR v2 small_45m` (`~41.3M`) + `MDMMeasureRefiner` (`~2.75M`, 4-layer non-causal GQA cross-attn denoiser sharing `tok_emb` & `lm_head`) = **`~44.1M` params** (`~23.8 MB` INT4 / `~32.8 MB` Hybrid FP16/INT4, `< 55 MB`)
- **Datasets (100% permissive):**
  - `gregoreesmaa/sheet-music-converter-omr` → `datasets/iter10_5_proxy_2k.tar` (`2,000` train + `64` val full pages)
  - `gregoreesmaa/sheet-music-converter-omr` → `datasets/suite31_hw_eval.tar` (45-piece `suite31_hw` OOD benchmark: 31 printed + 14 real historical HW, `clean` + `photo` = 90 pages)

## Moonshot Thesis (Why This Could Break the Causal AR Ceiling on Historical HW)
In left-to-right autoregressive `bekern` decoding, a single misread handwritten clef, accidental, or smudged beat permanently corrupts every downstream token in the measure — the causal decoder can never look *rightward* to the barline, beaming group, or harmonic cadence to disambiguate a messy handwritten notehead.

`MDM-MeasureRefiner` pairs `UniOMR v2` with a compact **4-layer Bidirectional Masked-Diffusion Denoiser (`2.75M` params)**:
1. `UniOMR v2` generates a fast causal FSA draft + per-token confidence scores `p_max(i)`.
2. Uncertain tokens (`p_max < tau`, capped at `35%`, plus rhythm-anomalous measures) are replaced with `[MASK]`.
3. `MDM-MeasureRefiner` runs `K=4` parallel **MaskGIT/MDLM cosine-unmasking steps** with **full bidirectional self-attention + visual cross-attention to `mem_final`**, seeing both left clef/key context and right barline/beam resolution simultaneously, followed by a fast `PrefixState` FSA validity projection.

## Primary Risk
Discrete masked diffusion on strict musical syntax can break `bekern` structural validity (`DecOK` drop) or overwrite correct rare accidentals if the confidence gate `tau` is miscalibrated.

## Quick Links
- Detailed Plan & Kill/Promote Gates: [PLAN.md](./PLAN.md)
- Standalone Colab L4 Runner: [run_colab.py](./run_colab.py)
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-26T07-40-i13-20-masked-diffusion-measure-denoiser.json`
- HF Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-26T07-40-i13-20-masked-diffusion-measure-denoiser.json`
