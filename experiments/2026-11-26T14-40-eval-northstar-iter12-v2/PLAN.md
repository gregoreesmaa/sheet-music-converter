# PLAN — `2026-11-26T14-40-eval-northstar-iter12-v2`

## 1. Platform Chosen & Rationale

- **Platform:** Local RTX 4080 SUPER (BF16 inference, `max_dim=1408`, `max_len=3584` per promoted `ov-35b` default) — or CPU fallback (`--device cpu`) for smoke tests.
- **Estimated Wall-Clock:** ~6–10 minutes on RTX 4080 SUPER (125 pages × ~3–4 s/page with MTP-4 + FSA grammar).
- **Cost:** `$0` (local GPU).
- **Dependency:** Launch ONLY after Iter 12 Stage 2 training finishes step 20,000 and frees the local RTX 4080 SUPER.

## 2. Identifiers for Follow-Up Queries

- **Local Out Dir:** `D:\tmp\northstar-eval-runs\iter12-v2\`
- **Predicted MusicXML Dir:** `D:\tmp\northstar-eval-runs\iter12-v2\musicxml\`
- **Final Report:** `D:\tmp\northstar-eval-runs\iter12-v2\report.json`
- **Per-Page Breakdown:** `D:\tmp\northstar-eval-runs\iter12-v2\per_page.json`
- **Hugging Face Upload Target:** `evals/2026-11-26T14-40-eval-northstar-iter12-v2_northstar.json`

## 3. Required Eval Data & Checkpoint

- **Manifest:** `D:\tmp\northstar-eval-pack\manifest.json` (`core_permissive`, `125` pages: `31 CDR + 31 ALS + 31 RWP + 28 HWM + 4 Negative Controls`; zero `CC-BY-NC`; validated with `omr_northstar.validate.validate_manifest` -> 0 errors).
- **Checkpoint:** `D:\tmp\mvd-runs\uni-omr-iter12-v2-stage2-20k\uni_omr_v2_ema_best.pt` (fallback: `uni_omr_v2_ema_latest.pt`).
- **MusicXML Converter:** `converter/assembly-serialize/target/release/assembly-serialize.exe bekern2musicxml` (MIT/permissive Rust binary).

## 4. Exact Launch Command & Hyperparameters

```powershell
# Full 125-page GPU eval (after Iter 12 Stage 2 finishes):
.\.venv\Scripts\python.exe -u experiments/2026-11-26T14-40-eval-northstar-iter12-v2/run_eval.py `
    --ckpt D:\tmp\mvd-runs\uni-omr-iter12-v2-stage2-20k\uni_omr_v2_ema_best.pt `
    --manifest D:\tmp\northstar-eval-pack\manifest.json `
    --out-dir D:\tmp\northstar-eval-runs\iter12-v2 `
    --device cuda --bf16 --max-dim 1408 --max-len 3584 --use-grammar --use-mtp-spec --resume
```

```powershell
# CPU smoke check (2 pages, zero GPU):
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe -u experiments/2026-11-26T14-40-eval-northstar-iter12-v2/run_eval.py `
    --ckpt D:\tmp\mvd-runs\uni-omr-iter12-v2-stage2-20k\uni_omr_v2_ema_best.pt `
    --manifest D:\tmp\northstar-eval-pack\manifest.json `
    --out-dir D:\tmp\northstar-eval-runs\iter12-v2-smoke `
    --device cpu --limit 2
```

## 5. Checkpointing & Resume Plan

- Every page writes `<out-dir>/musicxml/<page_id>.musicxml` atomically (`.tmp` -> `os.replace`) and flushes `<out-dir>/timing.json` after each page.
- Re-running with `--resume` skips completed pages and immediately resumes from the first unfinished page.
- Scoring (`eval_northstar_runner.py`) runs deterministically on CPU in `< 5 s` and can be re-run standalone on `<out-dir>/musicxml/` at any time.

## 6. Evaluation Gate & Decision Tree

- **Metrics:** Composite `NS` score (`[0, 100]`), `M_geom`, `Worst Stratum Q`, `CFR_max`, per-domain `Q` & `CFR` (`CDR`, `ALS`, `RWP`, `HWM`), `Δ_photo`, `median_wFPP`, `mean_NTSR`, `NUPR`, and Pillar 3B `NMHR` on 4 negative controls.
- **PROMOTE:** Beats all external baselines (`Audiveris`, `oemer`, `homr`) on composite `NS` and `HWM` / `RWP` strata -> record as canonical Iter 12 NorthStar baseline in `progress.md` and `todo.md`.
- **ITERATE:** Identify worst domain / worst 10 pages / highest error bucket (e.g., `NMHR > 0` or `ALS`/`HWM` collapse) -> spawn targeted Iter 13 fix experiment.
- **DISCARD:** Only if checkpoint is corrupt/mismatched; verify RAW vs EMA parity before discarding.
