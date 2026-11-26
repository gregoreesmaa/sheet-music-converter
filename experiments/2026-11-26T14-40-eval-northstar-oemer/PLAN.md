# PLAN — `2026-11-26T14-40-eval-northstar-oemer`

## 1. Platform Chosen & Rationale

- **Platform:** Local CPU (`$env:CUDA_VISIBLE_DEVICES=""`, ONNXRuntime CPU execution provider — zero GPU contention with RTX 4080 SUPER training).
- **Binary:** `.venv\Scripts\oemer.exe` (already installed in `.venv`; override with `--oemer-bin` for an isolated venv if desired).
- **Estimated Wall-Clock:** ~25–45 minutes for 125 pages on Ryzen 7 7800X3D CPU (`--timeout-s 180` per page).
- **Cost:** `$0`.

## 2. Identifiers for Follow-Up Queries

- **Local Out Dir:** `D:\tmp\northstar-eval-runs\oemer\`
- **Predicted MusicXML Dir:** `D:\tmp\northstar-eval-runs\oemer\musicxml\`
- **Final Report:** `D:\tmp\northstar-eval-runs\oemer\report.json`
- **Per-Page Breakdown:** `D:\tmp\northstar-eval-runs\oemer\per_page.json`

## 3. Required Eval Data

- **Manifest:** `D:\tmp\northstar-eval-pack\manifest.json` (`core_permissive`, `125` pages: `31 CDR + 31 ALS + 31 RWP + 28 HWM + 4 Negative Controls`).

## 4. Exact Launch Command

```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe -u experiments/2026-11-26T14-40-eval-northstar-oemer/run_eval.py `
    --manifest D:\tmp\northstar-eval-pack\manifest.json `
    --out-dir D:\tmp\northstar-eval-runs\oemer `
    --timeout-s 180 --resume
```

```powershell
# Optional 2-page CPU smoke check:
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe -u experiments/2026-11-26T14-40-eval-northstar-oemer/run_eval.py `
    --manifest D:\tmp\northstar-eval-pack\manifest.json `
    --out-dir D:\tmp\northstar-eval-runs\oemer-smoke `
    --timeout-s 180 --limit 2
```

## 5. Checkpointing & Resume Plan

- Each page writes `<out-dir>/musicxml/<page_id>.musicxml` atomically and flushes `<out-dir>/timing.json`.
- `--resume` skips completed pages; `eval_northstar_runner.py` can be re-run standalone at any time.

## 6. Evaluation Gate & Decision Tree

- **Metrics:** Full 4-layer `report.json` (`NS`, `CDR`/`ALS`/`RWP`/`HWM` strata, `Δ_photo`, `median_wFPP`, `NUPR`, `NMHR`).
- **PROMOTE / RECORD:** Record `oemer` as an official permissive open-source baseline row in our OMR-NorthStar leaderboard (`progress.md` / `OUTCOME.md`).
- **ITERATE:** If `oemer` crashes on specific high-res images or deskew failures, log exact failure rate (`parseable=False`) faithfully as part of `CFR` (do not hand-filter failures).
- **DISCARD:** Baseline benchmark only; no repo integration.
