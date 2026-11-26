# PLAN — `2026-11-26T14-40-eval-northstar-audiveris`

## 1. Platform Chosen & Rationale

- **Platform:** Local CPU + Docker Desktop (`docker 29.5.3`, zero GPU usage — safe to run while RTX 4080 SUPER is busy or idle).
- **Isolation:** Black-box Docker image (`audiveris-northstar:5.3.1`, built from `eclipse-temurin:21-jre` + official Audiveris release). Run with `--network none` and read-only input mount (`:ro`).
- **Estimated Wall-Clock:** ~15–25 minutes for 125 pages (CPU, `--timeout-s 120` per page).
- **Cost:** `$0`.

## 2. Identifiers for Follow-Up Queries

- **Docker Image:** `audiveris-northstar:5.3.1`
- **Local Out Dir:** `D:\tmp\northstar-eval-runs\audiveris\`
- **Predicted MusicXML Dir:** `D:\tmp\northstar-eval-runs\audiveris\musicxml\`
- **Final Report:** `D:\tmp\northstar-eval-runs\audiveris\report.json`
- **Per-Page Breakdown:** `D:\tmp\northstar-eval-runs\audiveris\per_page.json`

## 3. Required Eval Data

- **Manifest:** `D:\tmp\northstar-eval-pack\manifest.json` (`core_permissive`, `125` pages: `31 CDR + 31 ALS + 31 RWP + 28 HWM + 4 Negative Controls`).

## 4. Exact Launch Command

```powershell
$env:CUDA_VISIBLE_DEVICES=""
# Step 1 (one-time): build isolated Audiveris Docker image (strictly AGPL-3.0 sandboxed)
.\.venv\Scripts\python.exe -u experiments/2026-11-26T14-40-eval-northstar-audiveris/run_eval.py `
    --build-image --image audiveris-northstar:5.3.1 --skip-eval

# Step 2: run black-box Docker eval on all 125 pages
.\.venv\Scripts\python.exe -u experiments/2026-11-26T14-40-eval-northstar-audiveris/run_eval.py `
    --image audiveris-northstar:5.3.1 `
    --manifest D:\tmp\northstar-eval-pack\manifest.json `
    --out-dir D:\tmp\northstar-eval-runs\audiveris `
    --timeout-s 120 --resume
```

## 5. Checkpointing & Resume Plan

- Each page writes `<out-dir>/musicxml/<page_id>.mxl` (or `.musicxml` on clean empty-page rejection) atomically and updates `<out-dir>/timing.json`.
- `--resume` skips already-completed pages so interrupted runs pick up cleanly.

## 6. Evaluation Gate & Decision Tree

- **Metrics:** Full 4-layer `report.json` (`NS`, `CDR`/`ALS`/`RWP`/`HWM` strata, `Δ_photo`, `median_wFPP`, `NUPR`, `NMHR`).
- **PROMOTE / RECORD:** Publish Audiveris as an official external baseline row in our OMR-NorthStar comparison table (`progress.md` / `OUTCOME.md`).
- **ITERATE:** If Docker OOMs or times out on large high-DPI pages, tune container `--memory 6g` / `--timeout-s 180` and re-run with `--resume`.
- **DISCARD:** Never promote AGPL-3.0 code into our repo; benchmark artifact only.
