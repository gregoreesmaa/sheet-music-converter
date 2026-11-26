# `2026-11-26T07-40-i13-26-gqa-kv-heads-sweep`

**Title:** Iter 13 Exp 26 — Decoder GQA `n_kv_heads` Sweep (`1`, `2`, `4`, `8`) for Visual Cross-Attention Bottleneck Audit on `UniOMR v2 small_45m`
**Status:** `PROPOSED` (Iter 13 Proposals `batch_09.json`)
**Baseline Commit SHA:** `9303e51`
**Target Platform:** Google Colab `1× NVIDIA L4` (24 GB VRAM, native BF16, ~28–34 min total, ~2.6 CU)

---

## Summary & Hypothesis

In `GroupedQueryAttention` (`tmp-train/mvd/uni_omr_v2.py:1522-1615`), `UniOMRv2Config.small_45m()` sets `n_heads=8, n_kv_heads=2` (`4:1` GQA, `head_dim=48`, `kv_dim=96`) identically for **both** decoder causal self-attention and decoder visual cross-attention across all 8 decoder layers.

While `4:1` GQA is nearly lossless for text-only LLM self-attention, OMR decoder cross-attention is the sole bottleneck reading `K <= 3,072` 2D visual tokens for pitch height, accidentals, ledger lines, rhythm flags, articulation dots, and vertical system tracking simultaneously. Compressing all visual K/V projections into only 96 channels (`2 × 48`) may bottleneck dense polyphonic and handwritten page accuracy, whereas `n_kv_heads=4` (`2:1` GQA, `kv_dim=192`) adds minimal parameters (+~1.8M, still `< 35 MB` Hybrid INT4) and only 2× KV-cache RAM.

**Single-Variable 4-Arm Sweep (`400` steps/arm, `seed=42`, `Muon+AdamW`, `lr=3e-4`, `dim=1280`, BF16):**
1. **`kv1_mqa`**: `n_kv_heads=1` (8:1 MQA, `kv_dim=48`)
2. **`kv2_gqa4x`**: `n_kv_heads=2` (4:1 GQA, `kv_dim=96`, current default control)
3. **`kv4_gqa2x`**: `n_kv_heads=4` (2:1 GQA, `kv_dim=192`)
4. **`kv8_mha`**: `n_kv_heads=8` (1:1 full MHA, `kv_dim=384`)

**Hypothesis:** `kv4_gqa2x` (`n_kv_heads=4`) will deliver the best accuracy/KV-cache Pareto point, improving held-out `proxy_val` Token F1 by `>= +0.5 pp` over `kv2_gqa4x` while keeping Hybrid INT4 size `< 36 MB` and full-page FP16 KV-cache `< 16 MB`, whereas `kv1_mqa` will expose a sharp visual-readout cliff.

---

## Quick Links

- **Plan & Runbook:** [PLAN.md](./PLAN.md)
- **Standalone Colab Runner:** [run_colab.py](./run_colab.py)
- **Queue Proposal:** [batch_09.json](../iter13_proposals/batch_09.json)
- **Outcome:** `OUTCOME.md` *(written after Colab run completes)*
- **HF Private Hub Artifacts (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-26T07-40-i13-26-gqa-kv-heads-sweep.json`
  - Run Ledger: `runs/2026-11-26T07-40-i13-26-gqa-kv-heads-sweep.json`
