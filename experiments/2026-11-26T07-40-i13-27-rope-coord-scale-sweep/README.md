# `2026-11-26T07-40-i13-27-rope-coord-scale-sweep`

**Title:** Iter 13 Exp 27 — Aspect-Normalized 1D/2D-RoPE `rope_coord_scale` Sweep (`32`, `64`, `128`, `256`) for Spatial Frequency Aliasing Audit on `UniOMR v2 small_45m`
**Status:** `PROPOSED` (Iter 13 Proposals `batch_09.json`)
**Baseline Commit SHA:** `9303e51`
**Target Platform:** Google Colab `1× NVIDIA L4` (24 GB VRAM, native BF16, ~28–34 min total, ~2.6 CU)

---

## Summary & Hypothesis

In `AspectNormalizedRoPE1D` (`tmp-train/mvd/uni_omr_v2.py:505-560`), `AspectNormalizedRoPE2D` (`:562-698`), and `HybridAxialMergerEncoder._get_pos2d_grid` (`:1143-1185`), normalized spatial coordinates `y_norm, x_norm in [0, 1]` are multiplied by `rope_coord_scale = 128.0` before multiplying by `inv_freq` (where the highest rotary frequency has `inv_freq[0] = 1.0`).

At `1280px` full-page resolution, the post-merger `/16` global encoder grid has height `h_p ≈ 80` (and `h_p ≈ 50–60` on landscape/short pages, or `h_p = 12` on `192px` strips!). With `rope_coord_scale = 128.0`, the per-patch rotary phase step on the highest frequency band is:
- `128 / 79 = 1.62 rad/patch` (`> 92°` per single patch on portrait full pages),
- `128 / 55 = 2.33 rad/patch` (approaching Nyquist $\pi$ aliasing on shorter pages),
- `128 / 11 = 11.6 rad/patch` (complete multi-cycle aliasing on strips!).

Lowering `rope_coord_scale` to `64.0` or `32.0` keeps adjacent `/16` staff/chord patches strictly sub-Nyquist (`0.81` or `0.41 rad/patch`), giving smooth, non-aliased relative positional attention across adjacent staff lines and chord stacks, while `256.0` tests the over-aliased regime (`3.24 rad/patch > pi`).

**Single-Variable 4-Arm Sweep (`400` steps/arm, `seed=42`, `Muon+AdamW`, `lr=3e-4`, `dim=1280`, BF16):**
1. **`scale_32`**: `rope_coord_scale=32.0` (`0.41 rad/patch` at `/16`)
2. **`scale_64`**: `rope_coord_scale=64.0` (`0.81 rad/patch` at `/16`)
3. **`scale_128`**: `rope_coord_scale=128.0` (`1.62 rad/patch` at `/16`, current default control)
4. **`scale_256`**: `rope_coord_scale=256.0` (`3.24 rad/patch` at `/16`, above Nyquist $\pi$)

**Hypothesis:** `scale_64` (matching the native `/16` grid size) will outperform `scale_128` by `>= +0.5 pp` EMA Token F1 and lower `SER` with **0% parameter, VRAM, or runtime cost**, whereas `scale_256` will degrade due to high-frequency rotary phase wrap-around.

---

## Quick Links

- **Plan & Runbook:** [PLAN.md](./PLAN.md)
- **Standalone Colab Runner:** [run_colab.py](./run_colab.py)
- **Queue Proposal:** [batch_09.json](../iter13_proposals/batch_09.json)
- **Outcome:** `OUTCOME.md` *(written after Colab run completes)*
- **HF Private Hub Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-26T07-40-i13-27-rope-coord-scale-sweep.json`
  - Run Ledger: `runs/2026-11-26T07-40-i13-27-rope-coord-scale-sweep.json`
