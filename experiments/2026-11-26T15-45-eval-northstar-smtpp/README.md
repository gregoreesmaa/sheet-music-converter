# `2026-11-26T15-45-eval-northstar-smtpp` — SMT++ (`PRAIG/smt-fp-grandstaff`) on OMR-NorthStar

- **Status:** `PLANNED`
- **Baseline Commit SHA:** `e4e71bf`
- **Target System:** **SMT++** (`PRAIG/smt-fp-grandstaff`, MIT license, ungated — end-to-end full-page Sheet Music Transformer trained on FP-GrandStaff, ConvNeXt encoder + 8-layer Transformer decoder generating Humdrum `**kern`)
- **Benchmark:** Self-contained 125-page permissive OMR-NorthStar eval pack (`datasets/northstar_eval_pack_125p.tar.gz` on HF Private Hub: `31 CDR + 31 ALS + 31 RWP + 28 HWM + 4 Negative Controls`)

## Summary & Hypothesis

Evaluate **`PRAIG/smt-fp-grandstaff`** (the highest-download public ungated full-page SMT++ checkpoint on Hugging Face, `2,448` downloads, MIT) head-to-head on all 125 pages of our permissive OMR-NorthStar eval pack (`CDR`, `ALS`, `RWP`, `HWM` + 4 Pillar 3B Negative Controls).

SMT++ is the strongest published end-to-end full-page neural OMR baseline for grand-staff pianoform scores (`**kern` output, converted to MusicXML via `music21`). Measuring its full 4-layer `NS` scorecard tests:
1. How well a grand-staff-specialized full-page `**kern` transformer generalizes across non-piano/multi-layout `CDR` scores, archival scans (`ALS`), phone photos (`RWP`), and historical handwritten manuscripts (`HWM`).
2. Whether `**kern` → `music21` → MusicXML conversion introduces parse failures (`CFR`) or spine-alignment drift on out-of-distribution layouts.
3. Its negative-control hallucination rate (`NMHR`) on blank/text/non-music pages.

## Quick Links

- Plan & Modal L4 Launch Commands: [PLAN.md](./PLAN.md)
- Cloud L4 Eval Script (Modal + Colab compatible): [run_colab.py](./run_colab.py)
- Outcome: `OUTCOME.md` *(written immediately upon completion)*
- Modal Generic Runner: [modal_runner.py](../overnight_queue/modal_runner.py)
