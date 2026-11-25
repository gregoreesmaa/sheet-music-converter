# `2026-11-26T01-30-academic-gevt-vbkern-mr` — Gauge-Equivariant Visual Tokenization (`V-bekern`) & 2-Step Bidirectional Masked Refinement

**Status:** `RUNNING`
**Baseline Commit SHA:** `164852f`
**Platform:** Google Colab `L4` (24 GB VRAM, BF16 native, session `acad-c-gevt-vbkern`)

## Summary & Hypothesis

Standard semantic `bekern` forces a neural OMR decoder to perform long-range symbolic state transport inside attention weights: the exact same visual notehead on the 2nd staff line must be emitted as `G 4` in treble clef, `B 2` in bass clef, `A 3` in alto (`C3`) clef, or `F 3` in tenor (`C4`) clef, and every diatonic sharp/flat must be re-emitted on every note from the key signature.

**Hypothesis (Academic Experiment C):**
1. **Bijective Gauge-Equivariant Visual Tokenization (`V-bekern`):** Compiling semantic `<STEP> <OCT> [<ALTER>]` into staff-relative visual position `sp<k>` (`sp+0` = bottom staff line) + visually drawn accidentals ONLY (`acc-2..acc+2`), paired with a zero-parameter deterministic gauge-lifter `vbkern_to_bekern`, achieves:
   - **100.000% lossless round-trip** across all 45 `suite31_hw` GT pieces (`57,485 / 57,485` exact tokens).
   - **26.7% shorter target sequences** (`57,485 -> 42,156` tokens), reducing decoder exposure bias and autoregressive latency.
   - **Exact clef & key-signature equivariance**, eliminating gauge-transport errors on rare/unseen C-clefs and heavy key signatures.
2. **Lightweight 2-Step Bidirectional Masked Refinement (`MaskedRefinerHead`, `581K` params `< 0.8M`):** Plugs onto `UniOMRv2Net` decoder hidden states via a residual bottleneck (`d_ref=160`, 2 bidirectional layers) sharing `lm_head`, refining low-confidence spans in 2 parallel steps.

## Quick Links
- Execution & decision plan: [`PLAN.md`](PLAN.md)
- Colab runner script: [`run_colab.py`](run_colab.py)
- Core implementation: [`tmp-train/mvd/vbkern.py`](../../tmp-train/mvd/vbkern.py)
- CPU unit & 45-piece round-trip tests: [`tmp-train/mvd/test_vbkern.py`](../../tmp-train/mvd/test_vbkern.py)
- HF Hub Eval JSON: `gregoreesmaa/sheet-music-converter-omr` -> `evals/2026-11-26T01-30-academic-gevt-vbkern-mr.json`
- HF Hub Run Ledger: `gregoreesmaa/sheet-music-converter-omr` -> `runs/2026-11-26T01-30-academic-gevt-vbkern-mr.json`
- Outcome report: `OUTCOME.md` (written immediately upon completion)
