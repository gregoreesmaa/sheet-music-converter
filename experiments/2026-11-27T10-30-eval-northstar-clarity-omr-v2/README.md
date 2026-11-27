# `2026-11-27T10-30-eval-northstar-clarity-omr-v2` — Clarity-OMR (`clquwu/Clarity-OMR`) on OMR-NorthStar (`db12da9`)

- **Status:** `PLANNED`
- **Baseline Commit SHA:** `cfc94aa`
- **Scorer Version:** `gregoreesmaa/omr-northstar` (`main`, `>= db12da9`, freshly cloned & prepended to `sys.path`)
- **Target System:** **Clarity-OMR** (code: `https://github.com/clquwu/Clarity-OMR`, **GPL-3.0**; weights: HF `clquwu/Clarity-OMR`, **CC-BY-SA-4.0**, **ungated** — Stage-A YOLO staff detector `yolo.pt` + Stage-B DaViT-Base encoder / RoPE+DoRA Transformer decoder `model.safetensors` with 487-token vocabulary & Grammar FSA beam search width 5 → MusicXML)
- **Benchmark:** Self-contained 125-page permissive OMR-NorthStar eval pack (`datasets/northstar_eval_pack_125p.tar.gz` on HF Private Hub: `31 CDR + 31 ALS + 31 RWP + 28 HWM + 4 Negative Controls`)

## Summary & Hypothesis

Re-evaluate **`clquwu/Clarity-OMR`** (#2 contender) on all 125 pages of our permissive OMR-NorthStar eval pack using the newly audited **`omr-northstar` `db12da9`** scorer, and archive all 125 raw predicted `.musicxml` files to HF Private Hub (`evals/2026-11-27T10-30-eval-northstar-clarity-omr-v2_musicxml.tar.gz`) so future scorer updates can rescore Clarity-OMR in ~4 seconds on CPU.

Key upgrades over `2026-11-26T21-50-eval-northstar-clarity-omr` (v1):
1. **Audited `db12da9` scorer:** Clones/pulls `https://github.com/gregoreesmaa/omr-northstar.git` and prepends it to `sys.path` before importing `omr_northstar`, eliminating stale scorer drift from the HF pack snapshot.
2. **Persistent raw MusicXML archive:** Packages all 125 `.musicxml` outputs into `_musicxml.tar.gz` and uploads it alongside the canonical 4-layer report and per-page breakdown.

## Quick Links

- Plan & Modal L4 Launch Commands: [PLAN.md](./PLAN.md)
- Cloud L4 Eval Script (Modal + Colab compatible): [run_colab.py](./run_colab.py)
- Outcome: `OUTCOME.md` *(written immediately upon completion)*
- Modal Generic Runner: [modal_runner.py](../overnight_queue/modal_runner.py)
- HF Eval Output: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-27T10-30-eval-northstar-clarity-omr-v2.json`
- HF MusicXML Archive: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-27T10-30-eval-northstar-clarity-omr-v2_musicxml.tar.gz`
