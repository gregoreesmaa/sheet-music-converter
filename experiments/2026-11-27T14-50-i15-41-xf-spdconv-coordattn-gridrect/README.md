# `2026-11-27T14-50-i15-41-xf-spdconv-coordattn-gridrect`

- **Status:** `PLANNED` (P0 `wave=0` cross-field architectural proxy)
- **Short ID:** `i15-41`
- **Baseline Commit SHA:** `6055c49`

## Summary & Hypothesis
Tests a **cross-field early-vision front-end (`spd_coord_rect`)** in `UniOMRv2Net(edge85m, ax0_gl6, d=512)` against the stock `AnisotropicMusicBackbone` control (`ctrl`) under a controlled 2-arm 600-step BF16 `muon+adamw` proxy on `iter10_5_proxy_2k.tar`.

Combines two proven cross-field primitives before the ViT encoder:
1. **PCB Automated Optical Inspection (`SPD-Conv` + 1D Coordinate Attention):** Replaces strided `3×3` downsampling in early backbone stages (`stage1` `/1→/2` and `stage2` `/2→/4`) with lossless Space-to-Depth (`PixelUnshuffle(2)` + non-strided `3×3` conv) so 1–2 px staff lines, ledger lines, augmentation dots, and accidentals are never aliased away; adds lightweight 1D horizontal/vertical Coordinate Attention (`CoordAttn2d`) after `stage2`, `stage3`, and `stage4` to preserve exact row/column spatial coordinates across full staves and barlines.
2. **Astrophysics / Cartography (In-Encoder Curvilinear Grid Rectification):** Inserts a tiny identity-initialized residual 1D vertical grid-rectification warp (`VerticalGridRectifier`, `F.grid_sample` + spatial smoothness penalty) on the `/8` feature map right before the ViT encoder to straighten curved, bowed, or tilted staves prior to 2×2 patch merging and 2D-RoPE.

All experimental code lives strictly inside `run_colab.py` (`tmp-train/mvd/uni_omr_v2.py` untouched) and stays well below the `< 55.0 MB` packed Hybrid INT4 budget.

## Quick Links
- [PLAN.md](PLAN.md)
- [run_colab.py](run_colab.py)
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-27T14-50-i15-41-xf-spdconv-coordattn-gridrect.json`
- HF Partial Sync: `checkpoints/2026-11-27T14-50-i15-41-xf-spdconv-coordattn-gridrect/`
