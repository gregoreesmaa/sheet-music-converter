# `2026-11-26T07-40-i13-21-dual-stream-learned-ink-encoder` — Dual-Stream Raw-RGB + Differentiable Learned-Binarized-Ink Encoder (`DualInkStream`)

- **Experiment ID:** `2026-11-26T07-40-i13-21-dual-stream-learned-ink-encoder`
- **Proposal Slot:** `i13-21` (Iter 13 Risky Bets — Batch 07, Moonshot 3)
- **Status:** `PROPOSED`
- **Platform / GPU:** Google Colab `1× NVIDIA L4` (`24 GB` VRAM, BF16 native, ~42 min, ~3.5 CU)
- **Model Footprint:** `UniOMR v2 small_45m` (`~41.3M`) + `LearnedInkBinarizer` (`0.18M`) + `InkStreamEncoder` (`0.68M`) = **`~42.2M` params** (`~22.6 MB` INT4 / `~31.2 MB` Hybrid FP16/INT4, `< 55 MB`)
- **Datasets (100% permissive):**
  - `gregoreesmaa/sheet-music-converter-omr` → `datasets/iter10_5_proxy_2k.tar` (`2,000` train + `64` val full pages)
  - `gregoreesmaa/sheet-music-converter-omr` → `datasets/rism_hw_5707.tar` (`5,707` unlabeled real historical RISM manuscripts for photometric-invariance SSL)
  - `gregoreesmaa/sheet-music-converter-omr` → `datasets/suite31_hw_eval.tar` (45-piece `suite31_hw` OOD benchmark: 31 printed + 14 real historical HW, `clean` + `photo` = 90 pages)

## Moonshot Thesis (Why This Could Close the Archival Parchment / Bleed-Through Domain Gap)
Real historical manuscripts (RISM, Debussy, Mozart) suffer from yellowed parchment, severe verso ink bleed-through, iron-gall fading, foxing, and uneven lighting. Classical binarization (Otsu/Sauvola) destroys faint pencil/ledger lines and fails on shadows, so modern OMR abandoned it — leaving raw-RGB CNN stems vulnerable to photometric shift.

`DualInkStream` gives `UniOMR v2` the best of both worlds (`+0.86M` params, exact identity at step 0 via zero-init fusion gates):
1. **Differentiable Learned-Ink Binarizer (`0.18M`):** Combines multi-scale local adaptive contrast (`15×15` & `41×41` high-pass residuals) with a tiny residual U-Net to emit a canonical soft-ink probability canvas `I_ink ∈ [0, 1]^(1×H×W)`.
2. **Unlabeled RISM Photometric-Invariance SSL (`5,707` pages):** Trains `I_ink` directly on real historical manuscripts to produce invariant ink maps under synthetic parchment tint, bleed-through ghosts, contrast washout, and vignette lighting.
3. **Gated Dual-Stream FiLM + Residual Fusion (`0.68M`):** Encodes `I_ink` into `/4` (`P2`) and `/8` (`P3/P4`) anisotropic features and fuses them with raw-RGB `AnisotropicMusicBackbone` features via zero-initialized `tanh(g)` gates — giving photometric invariance without ever losing faint ink cues from the raw RGB stream.

## Primary Risk
Zero-init fusion gates may stay collapsed (`|tanh(g)| < 0.02`), or `I_ink` may amplify verso bleed-through into false noteheads / erase faint handwritten ledger lines.

## Quick Links
- Detailed Plan & Kill/Promote Gates: [PLAN.md](./PLAN.md)
- Standalone Colab L4 Runner: [run_colab.py](./run_colab.py)
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-26T07-40-i13-21-dual-stream-learned-ink-encoder.json`
- HF Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-26T07-40-i13-21-dual-stream-learned-ink-encoder.json`
