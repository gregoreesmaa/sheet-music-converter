# `2026-11-26T07-40-i13-14-photo-illum-flatten-autocrop` — Commercial-Inspired Grain-Safe Illumination Flattening & Desk Auto-Crop (`i13-14`)

- **Status:** `PROPOSED` (Iter 13 Competitor Scout 2 — Commercial Sheet Music Scanners, Google Colab `NVIDIA L4`, `BF16`)
- **Commercial Inspiration:** **Halbestunde**, **PlayScore 2**, **Newzik**, and **Soundslice** phone-photo preprocessing. Unlike naive CLAHE (which amplified paper grain and hurt `hw_photo` in `ov-36`) or Hough deskew (which clamped on dark desk borders in `ov-36`), commercial mobile scanners use **low-frequency background division** (removing broad phone shadows and yellow gradients without boosting high-frequency grain) plus **page-boundary auto-cropping** (reclaiming the 10–25% of pixels wasted on dark table borders).
- **Baseline Checkpoint:** `HF_CKPT_PATH` (default `checkpoints/iter10_soup_baseline/uni_omr_soup.pt`; auto-compatible with `UniOMR v2` checkpoints) evaluated at canonical `max_dim=1280`, `max_len=3840` on all 45 `suite31_hw` pieces (`clean` + `photo` = 90 pages).
- **Single Variable Tested:** Grain-safe phone-photo image preprocessing prior to `1280px` resize:
  1. `baseline`: Raw RGB resize to `1280px`.
  2. `bg_divide_soft_a50`: Low-frequency Gaussian background division (`sigma = 0.04 * max_dim`, blended at `alpha=0.50`, white-point `p98`).
  3. `bg_divide_full_a85`: Stronger low-frequency background division (`alpha=0.85`, white-point `p98`).
  4. `desk_autocrop_safe`: 4-edge dark-desk border detection & safe auto-crop (trims only outer bands with dark desk luminance, capped at `<= 18%` per side, preserving `>= 2%` quiet safety margin; no-op on clean white pages).
  5. `bg_soft_plus_autocrop`: Combined `desk_autocrop_safe` + `bg_divide_soft_a50`.

## Hypothesis
Stripping dark desk margins reclaims 10–25% effective staff resolution and stops `InkTokenRouter` from spending visual tokens on table borders, while low-frequency background division equalizes overhead phone shadows without amplifying paper grain. Together they will lift `printed_photo` and `hw_photo` Strict Note F1 by `>= +0.008` with `< 2%` preprocessing latency (`~10 ms/page`) and zero regression on `clean`.

## Quick Links
- Execution Plan, Arms & Decision Tree: [PLAN.md](./PLAN.md)
- Standalone Colab L4 Runner: [run_colab.py](./run_colab.py)
- HF Eval Target: `gregoreesmaa/sheet-music-converter-omr` -> `evals/2026-11-26T07-40-i13-14-photo-illum-flatten-autocrop.json`
- HF Run Ledger: `gregoreesmaa/sheet-music-converter-omr` -> `runs/2026-11-26T07-40-i13-14-photo-illum-flatten-autocrop.json`
