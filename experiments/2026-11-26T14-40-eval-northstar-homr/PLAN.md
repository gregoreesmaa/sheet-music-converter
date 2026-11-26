# PLAN — `2026-11-26T14-40-eval-northstar-homr`

## 1. Platform Chosen & Rationale

- **Platform:** Local CPU (`$env:CUDA_VISIBLE_DEVICES=""`, ONNXRuntime CPU in an isolated venv `D:\tmp\venvs\homr` so `homr`'s pinned dependencies never touch our main `.venv` PyTorch 2.14.1 CUDA 12.6 install).
- **Estimated Wall-Clock:** ~20–35 minutes for 125 pages on Ryzen 7 7800X3D CPU (`--timeout-s 180` per page).
- **Cost:** `$0`.
- **Optional SMT++ Extension:** `homr` outputs `.musicxml` directly; if evaluating `SMT++` (which outputs `**bekern` / `**kern`), route its tokens through `assembly-serialize.exe bekern2musicxml` into a separate `<pred-dir>` and score with `eval_northstar_runner.py`.

## 2. Identifiers for Follow-Up Queries

- **Isolated Venv:** `D:\tmp\venvs\homr\` (`homr` MIT package)
- **Local Out Dir:** `D:\tmp\northstar-eval-runs\homr\`
- **Predicted MusicXML Dir:** `D:\tmp\northstar-eval-runs\homr\musicxml\`
- **Final Report:** `D:\tmp\northstar-eval-runs\homr\report.json`
- **Per-Page Breakdown:** `D:\tmp\northstar-eval-runs\homr\per_page.json`

## 3. Required Eval Data

- **Manifest:** `D:\tmp\northstar-eval-pack\manifest.json` (`core_permissive`, `125` pages: `31 CDR + 31 ALS + 31 RWP + 28 HWM + 4 Negative Controls`).

## 4. Exact Launch Command

```powershell
$env:CUDA_VISIBLE_DEVICES=""
# Step 1 (one-time): create isolated venv and install homr (never touches main .venv)
.\.venv\Scripts\python.exe -u experiments/2026-11-26T14-40-eval-northstar-homr/run_eval.py `
    --install-venv D:\tmp\venvs\homr --skip-eval

# Step 2: run black-box homr eval on all 125 pages
.\.venv\Scripts\python.exe -u experiments/2026-11-26T14-40-eval-northstar-homr/run_eval.py `
    --homr-bin D:\tmp\venvs\homr\Scripts\homr.exe `
    --manifest D:\tmp\northstar-eval-pack\manifest.json `
    --out-dir D:\tmp\northstar-eval-runs\homr `
    --timeout-s 180 --resume
```

## 5. Checkpointing & Resume Plan

- Each page writes `<out-dir>/musicxml/<page_id>.musicxml` atomically and flushes `<out-dir>/timing.json`.
- `--resume` skips completed pages; `eval_northstar_runner.py` can be re-run standalone at any time.

## 6. Evaluation Gate & Decision Tree

- **Metrics:** Full 4-layer `report.json` (`NS`, `CDR`/`ALS`/`RWP`/`HWM` strata, `Δ_photo`, `median_wFPP`, `NUPR`, `NMHR`).
- **PROMOTE / RECORD:** Record `homr` as an official permissive open-source neural OMR baseline row in our OMR-NorthStar leaderboard (`progress.md` / `OUTCOME.md`).
- **ITERATE:** Compare `homr` vs UniOMR Iter 12 per-stratum (`CDR`, `ALS`, `RWP`, `HWM`) and per-tier (`T1..T5`) to identify any repertoire/tier where `homr`'s staff-level recognizer outperforms our full-page decoder.
- **DISCARD:** Baseline benchmark only; no repo integration.
