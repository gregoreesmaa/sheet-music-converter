# `2026-11-26T01-38-ov-30-cont-header-omission-sweep` — Continuation-Header Omission Sweep (`omit_cont_header_prob`)

- **Status:** `QUEUED` (Overnight Wave 6, ready for Google Colab `L4`)
- **Baseline Commit SHA:** `164852f`
- **Model:** `UniOMR v2 small_45m` (`~43M` params, `d_model=384`, 2 axial + 4 global + 8 GQA dec, MTP-4)
- **Datasets (100% permissive):**
  - `gregoreesmaa/sheet-music-converter-omr` → `datasets/iter10_5_proxy_2k.tar` (`2,000` train + `64` val full pages, 0 `suite31_hw` overlap)
  - `gregoreesmaa/sheet-music-converter-omr` → `datasets/suite31_hw_eval.tar` (45-piece `suite31_hw` OOD benchmark: 31 printed + 14 real HW, `clean` + `photo` = 90 pages)

## Summary & Hypothesis
Historical handwritten manuscripts (Mozart, Debussy, Chopin, RISM autographs) routinely omit redundant clefs and key signatures at the beginning of continuation systems (Systems 2+), expecting the reader to carry forward the active clef and key from System 1. By contrast, engraved synthetic training data (MuseScore/Verovio) always prints full clef + key headers on every system, causing `UniOMR` to over-rely on local left-edge header glyphs and misread or drop clefs/pitches on continuation systems of real manuscripts.

This 3-arm controlled sweep isolates on-the-fly **continuation-header visual omission probability `omit_cont_header_prob ∈ {0.0, 0.35, 0.60}` on Systems 2+** during training: for each continuation system (`s_idx >= 1`) selected with probability `p`, the left-edge header strip (`5–9%` of active system width) is cleanly erased with local paper fill while keeping System 1's header and the full ground-truth bekern sequence intact—forcing the decoder to carry forward clef/key state across `<sys>` boundaries (`600` steps per arm, `UniOMR v2 small_45m`, identical `seed=42`, `lr=3e-4`, `dim=1280`, BF16 on Colab `L4`).

**Hypothesis:** Training with `omit_cont_header_prob = 0.35` (or `0.60`) will dramatically improve continuation-system Clef/Key F1 and Pitch-only F1 on the continuation-header-omitted `proxy_val` stress test and boost `suite31_hw` real handwritten (`hw_clean`, `hw_photo`) Pitch F1 and Strict Note F1 without regressing standard printed scores.

## Quick Links
- Colab Launcher: [run_colab.py](./run_colab.py)
- Plan & Decision Tree: [PLAN.md](./PLAN.md)
- Queue Manifest: [batch_06.json](../overnight_queue/batch_06.json)
- Outcome: `OUTCOME.md` *(written immediately upon completion)*
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-26T01-38-ov-30-cont-header-omission-sweep.json`
- HF Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-26T01-38-ov-30-cont-header-omission-sweep.json`
