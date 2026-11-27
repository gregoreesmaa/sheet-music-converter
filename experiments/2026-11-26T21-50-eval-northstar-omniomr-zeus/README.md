# `2026-11-26T21-50-eval-northstar-omniomr-zeus` — OmniOMR Zeus (`OmniOMR/zeus`) on OMR-NorthStar

- **Status:** `COMPLETED` (`NS = 1.50`, `CDR = 10.13`, `ALS = 1.54`, `RWP = 2.74`, `HWM = 3.45`; see [OUTCOME.md](./OUTCOME.md))
- **Baseline Commit SHA:** `93be51e`
- **Target System:** **OmniOMR / Zeus** (code: `https://github.com/OmniOMR/zeus`, **MIT**, Charles University Prague / UFAL; snapshot: `zeus-camera-grandstaff-lmx-1.0-2024-02-12.model`, **CC-BY-SA-4.0**, ungated, TensorFlow 2.12 on Python 3.10 — predicts Linearized MusicXML `LMX` → MusicXML)
- **Benchmark:** Self-contained 125-page permissive OMR-NorthStar eval pack (`datasets/northstar_eval_pack_125p.tar.gz` on HF Private Hub: `31 CDR + 31 ALS + 31 RWP + 28 HWM + 4 Negative Controls`)

## Summary & Hypothesis

Evaluate **OmniOMR Zeus** (`zeus-camera-grandstaff-lmx-1.0-2024-02-12.model`, trained on Camera GrandStaff to emit Linearized MusicXML `LMX` → MusicXML) head-to-head on all 125 pages of our permissive OMR-NorthStar eval pack (`CDR`, `ALS`, `RWP`, `HWM` + 4 Pillar 3B Negative Controls).

Because Zeus is a staff/grandstaff-level recognizer (not a full-page layout detector) and requires Python 3.10 + TensorFlow 2.12, `run_colab.py` isolates `zeus` in a cached Python 3.10 `uv` venv (`/vol/venvs/zeus-py310`), slices each full page into system crops using a deterministic OpenCV horizontal staff-projection bridge (returning valid empty MusicXML on blank/non-music pages with zero staff bands), batch-transcribes all crops per page via `zeus predict --lmx --quiet-tf`, and concatenates per-system MusicXML into a single page MusicXML. Measuring its 4-layer `NS` scorecard tests:
1. How accurately LMX-based recognition performs on clean (`CDR`), archival (`ALS`), phone-photo (`RWP`), and handwritten (`HWM`) scores when paired with a lightweight system cropper.
2. Whether direct LMX → MusicXML decoding avoids `**kern`/`ABC` conversion bottlenecks or suffers LMX decoding failures (`CFR`).

## Quick Links

- Plan & Modal L4 Launch Commands: [PLAN.md](./PLAN.md)
- Cloud L4 Eval Script (Modal + Colab compatible): [run_colab.py](./run_colab.py)
- Outcome: `OUTCOME.md` *(written immediately upon completion)*
- Modal Generic Runner: [modal_runner.py](../overnight_queue/modal_runner.py)
- HF Eval Output: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-26T21-50-eval-northstar-omniomr-zeus.json`
