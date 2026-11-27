# PLAN — `2026-11-27T10-30-eval-northstar-clarity-omr-v2`

## 1. Platform Chosen & Rationale

- **Platform:** **Modal L4** (`24 GB VRAM`, `~$0.80/hr` within `$30/mo` free tier) via `experiments/overnight_queue/modal_runner.py` (also runs unmodified on Google Colab L4).
- **Model & License:**
  - Code: `https://github.com/clquwu/Clarity-OMR` (**GPL-3.0**) — benchmark evaluation only, never integrated into `sheet-music-converter`.
  - Weights: HF `clquwu/Clarity-OMR` (**CC-BY-SA-4.0**, **ungated**; `yolo.pt` + `model.safetensors` + `config.json`).
- **Why v2 Rerun:**
  1. Clones/updates `https://github.com/gregoreesmaa/omr-northstar.git` (`main`, `>= db12da9`) and prepends it to `sys.path` before importing `omr_northstar`, replacing the stale scorer snapshot bundled in `northstar_eval_pack_125p.tar.gz`.
  2. Archives all 125 raw predicted `.musicxml` files into `evals/2026-11-27T10-30-eval-northstar-clarity-omr-v2_musicxml.tar.gz` on HF Private Hub so any future `omr-northstar` scorer update can rescore Clarity-OMR in ~4 seconds on CPU without GPU re-inference.
- **Estimated Wall-Clock:** ~15–35 minutes for 125 pages on L4 GPU (Stage-A YOLO + Stage-B DaViT-Base beam-5 per staff + MusicXML export + `omr_northstar` `db12da9` scoring).
- **Estimated Cost:** `~$0.20–$0.45` on Modal L4 free credit.

## 2. Identifiers for Follow-Up Queries

- **Experiment ID:** `2026-11-27T10-30-eval-northstar-clarity-omr-v2`
- **Scorer Repo:** `https://github.com/gregoreesmaa/omr-northstar.git` (`main`, `>= db12da9`)
- **HF Dataset Input:** `gregoreesmaa/sheet-music-converter-omr` → `datasets/northstar_eval_pack_125p.tar.gz`
- **HF Eval Output (canonical 4-layer report):** `evals/2026-11-27T10-30-eval-northstar-clarity-omr-v2.json`
- **HF Per-Page Breakdown:** `evals/2026-11-27T10-30-eval-northstar-clarity-omr-v2_per_page.json`
- **HF Raw MusicXML Archive (125 pages):** `evals/2026-11-27T10-30-eval-northstar-clarity-omr-v2_musicxml.tar.gz`
- **HF Run Ledger:** `runs/2026-11-27T10-30-eval-northstar-clarity-omr-v2.json`

## 3. Exact Modal L4 Launch Commands

```powershell
# Detached Modal L4 launch (recommended — safe against local disconnects):
.\.venv\Scripts\modal.exe run --detach experiments/overnight_queue/modal_runner.py `
  --exp-id 2026-11-27T10-30-eval-northstar-clarity-omr-v2 `
  --script-path experiments/2026-11-27T10-30-eval-northstar-clarity-omr-v2/run_colab.py `
  --hf-eval-target evals/2026-11-27T10-30-eval-northstar-clarity-omr-v2.json `
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
2. Clones/updates `https://github.com/gregoreesmaa/omr-northstar.git` (`main` / `>= db12da9`) to `/tmp/omr-northstar` and prepends it to `sys.path` before importing `omr_northstar` (verified via `scoring.__file__` assertion).
3. Clones `https://github.com/clquwu/Clarity-OMR.git` (`--depth 1`) to `/tmp/Clarity-OMR` and pre-resolves `yolo.pt` + `model.safetensors` from HF `clquwu/Clarity-OMR` once.
4. For each page (resume-safe via `/vol/runs/2026-11-27T10-30-eval-northstar-clarity-omr-v2/timing.json` + atomic `<page_id>.musicxml`), wraps the page image into a native-resolution 1-page 300-DPI PDF via PyMuPDF (`fitz`) and runs `python omr.py <page.pdf> -o <tmp.musicxml> --beam-width 5 --device cuda --pdf-dpi 300` with a 300s per-page timeout.
5. Scores all 125 pages using the audited `omr_northstar` (`>= db12da9`) stdlib scorer, archives all 125 `.musicxml` predictions into `2026-11-27T10-30-eval-northstar-clarity-omr-v2_musicxml.tar.gz`, and uploads the validated report + per-page breakdown + MusicXML tarball + run ledger to HF Private Hub.

## 5. Evaluation Gate & Decision Tree

- **RECORD / BASELINE:** Update Clarity-OMR (#2 contender) in the apples-to-apples `omr-northstar db12da9` leaderboard (`OUTCOME.md`, `progress.md`) and run `learn-from-northstar` forensic comparison against UniOMR.
- **FAITHFUL FAILURE LOGGING:** Never hand-filter timeouts or MusicXML export errors; log exact `parseable=False` count as part of `CFR` and record per-page status in `runs/<exp_id>.json`.
- **DISCARD (Code):** Evaluation baseline only (GPL-3.0); zero code/weights copied into `sheet-music-converter`.
