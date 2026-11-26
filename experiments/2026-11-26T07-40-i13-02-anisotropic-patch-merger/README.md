# `2026-11-26T07-40-i13-02-anisotropic-patch-merger`

**Title:** Iter 13 Academic Frontier Exp 02 — Vertical-Preserving Anisotropic Patch Merger (`1x4` / `1x2` vs `2x2`)
**Status:** `PROPOSED` (Iter 13 Academic Batch 01 — OMR & Document VLM Vision-to-Sequence Frontier)
**Baseline Commit SHA:** `9303e51`
**Target Platform:** Google Colab `L4` (24 GB VRAM, BF16 native, ~30–34 min total, ~2.6 CU)

---

## Summary & Hypothesis

`UniOMR v2` adopts **Qwen2.5-VL**'s isotropic `2x2` MLP `PatchMerger2x2` (`Bai et al., arXiv:2502.13923`, Feb 2025) after the `/8` backbone+axial encoder, producing `/16 y × /16 x` visual tokens before `InkTokenRouterV2`. However, **Sheet Music Transformer / SMT++** (*Ríos-Vila et al., ICDAR 2024 / arXiv:2405.12105*) and **Pix2Struct** (*Lee et al., ICML 2023*) highlight that sheet music has extreme **vertical pitch sensitivity** relative to horizontal spacing: at `1280px` page height, a 5-line staff is only `40–55 px` tall (`4–5.5 px` per diatonic step), so `/16` vertical downsampling collapses **3–4 diatonic pitch positions into a single visual token** — a Nyquist pitch-aliasing bottleneck on dense multi-staff printed and handwritten manuscripts (`suite31_hw`).

Because `InkTokenRouterV2` and `AspectNormalizedRoPE2D` are already 100% aspect-agnostic, replacing `2x2` merging with **anisotropic horizontal-first merging** (`1x4` or `1x2`) doubles vertical pitch resolution (`/8 y`) while keeping the `<55 MB` edge budget intact:
1. `merger_2x2_control`: `2×2` isotropic (`/16 y × /16 x`, 4× compression — exact current v2 baseline).
2. `merger_1x4_pitch_hi`: `1×4` anisotropic (`/8 y × /32 x`, **identical 4× compression, identical token count, identical parameter count**, 2× finer vertical pitch resolution!).
3. `merger_1x2_pitch_hi`: `1×2` anisotropic (`/8 y × /16 x`, 2× compression, 2× finer vertical pitch resolution + full `/16` horizontal resolution for tight chords/accidentals).
4. `merger_1x2_matched_budget`: `1×2` anisotropic (`/8 y × /16 x`) with `keep_ratio=0.20, max_keep_ratio=0.375` so the routed memory token count `k` matches `merger_2x2_control` (cleanly isolating vertical resolution from token budget).

**Hypothesis:** Preserving `/8` vertical pitch resolution via `1x4` or `1x2` anisotropic patch merging will reduce vertical pitch aliasing on dense staves and lift held-out full-page `proxy_val` Token F1 by `>= +0.6 pp` (and lower SER) over isotropic `2x2` merging at zero extra parameter cost.

---

## Quick Links

- **Plan & Runbook:** [PLAN.md](./PLAN.md)
- **Standalone Colab Runner:** [run_colab.py](./run_colab.py)
- **Outcome:** `OUTCOME.md` *(written after Colab run completes)*
- **HF Private Hub Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-26T07-40-i13-02-anisotropic-patch-merger.json`
  - Run Ledger: `runs/2026-11-26T07-40-i13-02-anisotropic-patch-merger.json`
