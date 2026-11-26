# `2026-11-26T07-40-i13-25-axial-global-layer-split`

**Title:** Iter 13 Exp 25 — Hybrid Encoder `n_axial_layers` vs `n_global_layers` Iso-Depth Split Sweep on `UniOMR v2 small_45m`
**Status:** `PROPOSED` (Iter 13 Proposals `batch_09.json`)
**Baseline Commit SHA:** `9303e51`
**Target Platform:** Google Colab `1× NVIDIA L4` (24 GB VRAM, native BF16, ~30–36 min total, ~2.8 CU)

---

## Summary & Hypothesis

In `HybridAxialMergerEncoder` (`tmp-train/mvd/uni_omr_v2.py:1078-1324`), `UniOMRv2Config.small_45m()` currently allocates a **2:4 split** (`n_axial_layers=2` at high-res `/8` with `axial_ffn_mult=2.0`, followed by 2×2 `PatchMerger2x2` `/8 -> /16` + `InkTokenRouterV2`, followed by `n_global_layers=4` at `/16` with `ffn_mult=4.0`).

Because musical notation is strongly axis-aligned (horizontal 5-line staves, beams, ties, slurs; vertical chord stacks, stems, barlines), the `/8` axial layers are the only attention layers that see the full unmerged high-resolution page grid before 4× spatial compression and token pruning. Allocating only 2 axial layers (with half-width FFN) may under-process high-res staff/chord geometry before irreversible merging, whereas shifting 1–2 layers from global `/16` to axial `/8` preserves total encoder depth (`6` layers) at nearly identical parameter count.

**Single-Variable Iso-Depth 4-Arm Sweep (`400` steps/arm, `seed=42`, `Muon+AdamW`, `lr=3e-4`, `dim=1280`, BF16):**
1. **`ax1_gl5`**: `n_axial_layers=1, n_global_layers=5` (global-heavy)
2. **`ax2_gl4`**: `n_axial_layers=2, n_global_layers=4` (current default control)
3. **`ax3_gl3`**: `n_axial_layers=3, n_global_layers=3` (balanced `/8` axial vs `/16` global)
4. **`ax4_gl2`**: `n_axial_layers=4, n_global_layers=2` (high-res axial-heavy)

**Hypothesis:** Shifting 1 layer to high-res axial (`ax3_gl3`) will improve held-out `proxy_val` Token F1 by `>= +0.5 pp` and `sys_count_exact` over the `ax2_gl4` default at `< 8%` throughput cost, whereas `ax1_gl5` will degrade fine-symbol accuracy (accidentals/dots/chords) due to premature `/8 -> /16` merging.

---

## Quick Links

- **Plan & Runbook:** [PLAN.md](./PLAN.md)
- **Standalone Colab Runner:** [run_colab.py](./run_colab.py)
- **Queue Proposal:** [batch_09.json](../iter13_proposals/batch_09.json)
- **Outcome:** `OUTCOME.md` *(written after Colab run completes)*
- **HF Private Hub Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-26T07-40-i13-25-axial-global-layer-split.json`
  - Run Ledger: `runs/2026-11-26T07-40-i13-25-axial-global-layer-split.json`
