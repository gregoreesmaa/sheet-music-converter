# `2026-11-27T07-46-i15-03-lyrics-textspotter-vs-deferred-bpe` — Lyrics & Text Bake-Off: Parallel `TextSpotterHead` vs Deferred `<sys_lyr>` Syllable BPE

- **Experiment ID:** `2026-11-27T07-46-i15-03-lyrics-textspotter-vs-deferred-bpe`
- **Short ID / Wave:** `i15-03` (Wave 13 — Representation & Text/Lyrics Pivot)
- **Status:** `QUEUED`
- **Baseline Commit SHA:** `e426c28`
- **Platform / GPU:** Google Colab `1× NVIDIA L4` (`24 GB` VRAM, BF16 native, est. ~75 min / ~5.5 CU)
- **Backbone & Footprint Budget:** `UniOMRv2Config.edge85m` (`d_model=512`, `10` dec layers, `< 55.0 MB` Hybrid FP16/INT4 hard ceiling):
  - **Arm A (`arm_a_textspotter_head`):** Frozen base encoder + decoder + `~1.8M` (`~1.15 MB` INT4) 2-layer `d=256` parallel `TextSpotterHead` cross-attending to `mem_post_merger` from notehead/bar decoder hidden states + unfrozen `encoder.router` text-ink saliency.
  - **Arm B (`arm_b_deferred_sys_lyr_bpe`):** Unified sequence with end-of-system bar-anchored `<sys_lyr>` + 256-token syllable BPE (`+256` vocab appended to `tok_emb` & `lm_head`).
- **Datasets (100% permissive):**
  - `gregoreesmaa/sheet-music-converter-omr` → `datasets/iter10_5_proxy_2k.tar` (vocal/leadsheet subset + XML-extracted lyrics/harmony/tempo sidecars)
  - `gregoreesmaa/sheet-music-converter-omr` → `datasets/suite31_hw_eval.tar` (45-piece `suite31_hw` OOD benchmark, 90 pages — vocal/lyrics/harmony/tempo subset + full `S_note` regression check)

## Summary & Hypothesis
In OMR-NorthStar evaluations, `S_expr` suffers a `-13.5 pp` structural penalty because `text = 0.000` (UniOMR currently emits zero lyrics, chord symbols/harmony, or tempo/direction text). Simply interleaving character-level text inline into the main bekern stream risks bloating sequence length past `max_len`, slowing autoregressive decode, and regressing vocal `S_note` F1.

This controlled 2-arm bake-off (`1,500` steps/arm, `seed=42`, BF16, identical vocal/leadsheet split) pits Pivot Strategist 3's two `< 55 MB`-compliant text/lyrics architectures head-to-head:
1. **`arm_a_textspotter_head`:** Keeps the base encoder & decoder weights frozen (guaranteeing `0.000` vocal `S_note` regression on frozen music tokens), unfreezes `encoder.router` so text-ink patches are routed into `mem_post_merger`, and trains a lightweight `~1.8M` (`~1.15 MB` INT4) 2-layer `d=256` parallel `TextSpotterHead` queried by notehead/bar decoder hidden states.
2. **`arm_b_deferred_sys_lyr_bpe`:** Extends the unified autoregressive decoder with `+256` syllable BPE tokens and emits a bar-anchored `<sys_lyr>` block at the end of each system (avoiding inline measure disruption while sharing full decoder context).

Both arms are judged on held-out `text` F1 (Promote gate: `>= 0.70`), vocal `S_note` delta (must be `>= -0.5 pp`), vocal decode latency, and exact on-disk Hybrid FP16/INT4 packed byte size on `edge85m` (`< 55.0 MB`).

## Quick Links
- Execution Plan & Decision Tree: [PLAN.md](./PLAN.md)
- Standalone Colab L4 Runner: [run_colab.py](./run_colab.py)
- Queue Entry Spec: [queue_entry.json](./queue_entry.json)
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-27T07-46-i15-03-lyrics-textspotter-vs-deferred-bpe.json`
- HF Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-27T07-46-i15-03-lyrics-textspotter-vs-deferred-bpe.json`
