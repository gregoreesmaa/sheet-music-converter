# `2026-11-26T07-40-i13-01-visual-causal-flow-encoder`

**Title:** Iter 13 Academic Frontier Exp 01 — DeepSeek-OCR 2 Visual Causal Flow Queries in `HybridAxialMergerEncoder`
**Status:** `PROPOSED` (Iter 13 Academic Batch 01 — OMR & Document VLM Vision-to-Sequence Frontier)
**Baseline Commit SHA:** `9303e51`
**Target Platform:** Google Colab `L4` (24 GB VRAM, BF16 native, ~30–34 min total, ~2.6 CU)

---

## Summary & Hypothesis

In `UniOMR v2` (`HybridAxialMergerEncoder`), `InkTokenRouterV2` sorts routed visual tokens in rigid 2D **raster-scan order** (`top-left -> bottom-right`), and `GlobalEncoderLayer` runs pure bidirectional self-attention over those raster-ordered tokens. On full-page polyphonic pianoform and multi-staff historical handwritten manuscripts (`suite31_hw`), musical reading order is non-raster (system-by-system, measure-by-measure, and vertically synchronized across grand-staff/quartet staves), forcing the GQA decoder's cross-attention to repeatedly jump across hundreds of raster-ordered visual tokens.

Inspired by **DeepSeek-OCR 2: Visual Causal Flow** (Wei, Sun & Li, *arXiv:2601.20552*, Jan 2026) and **GOT-OCR2.0** (Wei et al., *arXiv:2409.01704*, 2024), we test adding a compact bank of `M` learnable **Visual Causal Flow (VCF) query tokens** inside `GlobalEncoderLayer` with DeepEncoder-V2's **hybrid attention mask**:
- Visual tokens (+ registers) attend **bidirectionally** to all visual tokens (preserving global 2D spatial context) and never attend to flow queries.
- Causal flow queries attend to **all visual tokens** AND **causally to preceding flow queries**, progressively reordering 2D ink semantics into a 1D reading-order-aligned visual memory before the causal GQA decoder.

We evaluate 4 single-variable arms on `UniOMR v2 small_45m`:
1. `control_raster`: `M=0` (exact current `UniOMR v2` baseline).
2. `vcf_append_64`: `M=64` causal flow queries appended to `mem_levels` (`+0.05M` params, `<0.1 MB` INT4).
3. `vcf_append_128`: `M=128` causal flow queries appended to `mem_levels` (`+0.10M` params, `<0.1 MB` INT4).
4. `vcf_compress_256`: `M=256` causal flow queries *replacing* raw raster memory (pure DeepEncoder-V2 optical compression test).

**Hypothesis:** Appending `64–128` Visual Causal Flow queries (`vcf_append_64` / `vcf_append_128`) will give the GQA decoder a pre-ordered 1D reading-order scaffold alongside fine-grained 2D ink tokens, lifting held-out full-page `proxy_val` Token F1 by `>= +0.5 pp` and `sys_count_exact` by `>= +2.0 pp` at `<0.1 MB` edge footprint cost.

---

## Quick Links

- **Plan & Runbook:** [PLAN.md](./PLAN.md)
- **Standalone Colab Runner:** [run_colab.py](./run_colab.py)
- **Outcome:** `OUTCOME.md` *(written after Colab run completes)*
- **HF Private Hub Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-26T07-40-i13-01-visual-causal-flow-encoder.json`
  - Run Ledger: `runs/2026-11-26T07-40-i13-01-visual-causal-flow-encoder.json`
