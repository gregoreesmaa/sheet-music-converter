# `2026-11-26T01-38-ov-26-label-smoothing-struct-weight` — Overnight Exp 26: Label Smoothing × Structural Token Weight Sweep

- **Status:** `PLANNED` (`QUEUED` in `experiments/overnight_queue/batch_05.json`, Wave 6)
- **Baseline Commit SHA:** `164852f`
- **Model:** `UniOMR v2 small_45m` (`~43M` params)
- **Dataset (100% permissive):** `gregoreesmaa/sheet-music-converter-omr` → `datasets/iter10_5_proxy_2k.tar` (`2,000` train + `64` val full pages)

## Summary & Hypothesis
In full-page BeKern OMR, a single mispredicted **structural/header token** (`clef-*`, `key*`, `time-*`, `<sys>`, `grace`, `bar*`) invalidates the pitch or metric interpretation of an entire measure or system, whereas overconfident token logits from unsmoothed cross-entropy can amplify autoregressive decoding errors.

This experiment runs a controlled **3×2 full-factorial sweep** (`6` arms × `450` steps/arm, identical `seed=42`, `lr=3e-4`, `dim=1280`, BF16 on Colab `L4`) on `UniOMR v2 small_45m`:
- **`label_smoothing`:** `{0.0, 0.05, 0.10}`
- **`struct_tok_weight`:** `{1.0, 1.25}` (using the exact `UniOMRNet._get_struct_token_weights` definition: `w` on `clef-*`, `key*`, `time-*`, `<sys>`, `grace`, `bar*`; `0.5 * (1 + w)` on micro-rhythm/double-accidental tokens `tuplet-start-*`, `tuplet-stop`, `32nd`, `64th`, `a2`, `a-2`).

**Hypothesis:** Combining mild label smoothing (`0.05`) with structural token upweighting (`struct_tok_weight=1.25`) will improve structural-token F1 (`clef`/`key`/`time`/`bar`/`<sys>`), `<sys>` count exact match, and overall `proxy_val` Token F1 while improving probability calibration (lower ECE) compared to the unweighted `ls=0.0, w=1.0` baseline.

## Quick Links
- Colab Runner: [run_colab.py](./run_colab.py)
- Plan & Decision Tree: [PLAN.md](./PLAN.md)
- Queue Manifest: [batch_05.json](../overnight_queue/batch_05.json)
- Outcome: `OUTCOME.md` *(written immediately upon completion)*
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-26T01-38-ov-26-label-smoothing-struct-weight.json`
- HF Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-26T01-38-ov-26-label-smoothing-struct-weight.json`
