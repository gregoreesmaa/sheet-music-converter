# `2026-11-26T15-45-eval-northstar-qwen25vl` — Qwen2.5-VL-7B-Instruct (VLM → ABC → MusicXML) on OMR-NorthStar

- **Status:** `PLANNED`
- **Baseline Commit SHA:** `e4e71bf`
- **Target System:** **`Qwen/Qwen2.5-VL-7B-Instruct`** (Apache-2.0, ungated general-purpose Vision-Language Model, zero-shot prompted for ABC notation → `music21` → MusicXML)
- **Legato Access Note:** Live HF API check on `2026-11-26` confirmed **`guangyangmusic/legato`**, **`guangyangmusic/legato-small`**, and **`guangyangmusic/legato-2`** are all **gated (`403 GatedRepoError`, Llama 3.2 Vision license gate)**. Per protocol, this experiment evaluates the ungated **`Qwen/Qwen2.5-VL-7B-Instruct`** baseline on L4 (`24 GB VRAM`, BF16); once HF access to `guangyangmusic/legato` is approved, the exact same ABC → `music21` → `omr_northstar` pipeline can evaluate Legato directly.
- **Benchmark:** Self-contained 125-page permissive OMR-NorthStar eval pack (`datasets/northstar_eval_pack_125p.tar.gz` on HF Private Hub: `31 CDR + 31 ALS + 31 RWP + 28 HWM + 4 Negative Controls`)

## Summary & Hypothesis

Evaluate **`Qwen/Qwen2.5-VL-7B-Instruct`** zero-shot on all 125 pages of our permissive OMR-NorthStar eval pack (`CDR`, `ALS`, `RWP`, `HWM` + 4 Pillar 3B Negative Controls).

Frontier open-weights VLMs like Qwen2.5-VL-7B excel at document OCR and negative-control abstention (`NMHR`), making them an essential reference point on OMR-NorthStar:
1. Quantifies how far a general-purpose 7B VLM gets on dense polyphonic sheet music (`CDR`, `ALS`, `RWP`, `HWM`) via ABC transcription (`do_sample=False`, deterministic).
2. Tests whether VLM-generated ABC parses cleanly via `music21` or suffers structural rhythm/barline drift (`BSVR`, `s_struct`, `wFPP`).
3. Provides a strong negative-control (`NMHR`) baseline via explicit `NO_MUSIC` abstention prompting.

## Quick Links

- Plan & Modal L4 Launch Commands: [PLAN.md](./PLAN.md)
- Cloud L4 Eval Script (Modal + Colab compatible): [run_colab.py](./run_colab.py)
- Outcome: `OUTCOME.md` *(written immediately upon completion)*
- Modal Generic Runner: [modal_runner.py](../overnight_queue/modal_runner.py)
