# PLAN — `2026-11-26T15-45-eval-northstar-smtpp`

## 1. Platform Chosen & Rationale

- **Platform:** **Modal L4** (`24 GB VRAM`, native BF16/FP32, `~$0.80/hr` within `$30/mo` free tier) via `experiments/overnight_queue/modal_runner.py` (also runs unmodified on Google Colab L4).
- **Model:** `PRAIG/smt-fp-grandstaff` (MIT, ungated; `d_model=256`, `8` decoder layers, `maxh=2970`, `maxw=2100`, `maxlen=4353`).
  - *Alternative ungated PRAIG checkpoint:* `PRAIG/smt-fp-polish-scores` (can be tested via `SMT_MODEL_ID=PRAIG/smt-fp-polish-scores`). `PRAIG/smt-fp-mozarteum` is `gated=manual` and excluded.
- **Estimated Wall-Clock:** ~10–25 minutes for 125 full pages on L4 GPU (~5–12 s/page autoregressive decoding + `music21` `**kern` → MusicXML conversion + `omr_northstar` scoring).
- **Estimated Cost:** `~$0.15–$0.35` on Modal L4 free credit.

## 2. Identifiers for Follow-Up Queries

- **Experiment ID:** `2026-11-26T15-45-eval-northstar-smtpp`
- **HF Dataset Input:** `gregoreesmaa/sheet-music-converter-omr` → `datasets/northstar_eval_pack_125p.tar.gz`
- **HF Eval Output (canonical 4-layer report):** `evals/2026-11-26T15-45-eval-northstar-smtpp.json`
- **HF Per-Page Breakdown:** `evals/2026-11-26T15-45-eval-northstar-smtpp_per_page.json`
- **HF Run Ledger:** `runs/2026-11-26T15-45-eval-northstar-smtpp.json`

## 3. Exact Modal L4 Launch Commands

```powershell
# Detached Modal L4 launch (recommended — safe against local disconnects):
.\.venv\Scripts\modal.exe run --detach experiments/overnight_queue/modal_runner.py `
  --exp-id 2026-11-26T15-45-eval-northstar-smtpp `
  --script-path experiments/2026-11-26T15-45-eval-northstar-smtpp/run_colab.py `
  --hf-eval-target evals/2026-11-26T15-45-eval-northstar-smtpp.json `
  --gpu L4 `
  --timeout-secs 7200
```

```powershell
# Monitor running Modal app & logs:
.\.venv\Scripts\modal.exe app list
.\.venv\Scripts\modal.exe app logs <app-id> -f
```

## 4. Pipeline & Checkpointing Details

1. Downloads & unpacks `datasets/northstar_eval_pack_125p.tar.gz` to `/vol/datasets/northstar_eval_pack_125p/` (cached on Modal `omr-cache` Volume).
2. Clones `https://github.com/antoniorv6/SMT.git` (`--depth 1`) into `/tmp/SMT` and loads `SMTModelForCausalLM.from_pretrained("PRAIG/smt-fp-grandstaff")` on CUDA (automatically falling back to `use_flash_attn=False` if `flash_attn` is absent).
3. Resizes any page exceeding `(maxh=2970, maxw=2100)` with aspect-ratio preservation before `convert_img_to_tensor` grayscale preprocessing.
4. Runs autoregressive `**kern` inference per page, saves raw `<page_id>.krn`, converts `**kern` → MusicXML via `music21.converter.parseData(..., format="humdrum")`, and writes `<page_id>.musicxml` atomically to `/vol/runs/2026-11-26T15-45-eval-northstar-smtpp/musicxml/` (resume-safe).
5. Scores all 125 pages using the bundled `omr_northstar` stdlib scorer (`cer.score_musicxml_pair`, `scoring.build_report`, `stress.score_negative_control`, `validate.validate_report`) and uploads the validated report + per-page breakdown + run ledger to HF Private Hub.

## 5. Evaluation Gate & Decision Tree

- **RECORD / BASELINE:** Record `PRAIG/smt-fp-grandstaff` in the OMR-NorthStar competitor leaderboard (`OUTCOME.md`, `progress.md`) and run `learn-from-northstar` forensic comparison against UniOMR v2.
- **FAITHFUL FAILURE LOGGING:** Never hand-filter `music21` `**kern` parse errors or empty outputs; log exact `parseable=False` count as part of `CFR` and record conversion diagnostics in `runs/<exp_id>.json`.
- **DISCARD:** Baseline benchmark only; no upstream model integration.
