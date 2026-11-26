<#
.SYNOPSIS
    Canonical 2-stage Iter 13 UniOMR v2 launcher + automatic suite31_hw evaluation.
    Integrates all promoted cloud wins (ov-14, ov-21, ov-22, ov-24, ov-34b, ov-35b,
    i13-05, i13-10, i13-17) and all Iter-13 forensic P0/P1 bug fixes (f089d62).

.NOTES
    - PowerShell-safe (no `&&`, strict `$ErrorActionPreference = "Stop"`).
    - Local RTX 4080 SUPER (16 GB VRAM, BF16 native, CUTLASS EFFICIENT_ATTENTION).
    - Uses `--save-every-minutes 5.0` atomic checkpoints on both Stage 1 and Stage 2.
    - Pass `-SkipStage1`, `-SkipStage2`, or `-SkipEval` to resume cleanly.
#>
[CmdletBinding()]
param(
    [switch]$SkipStage1,
    [switch]$SkipStage2,
    [switch]$SkipEval,
    [string]$Stage1OutDir = "D:\tmp\mvd-runs\uni-omr-iter13-v2-stage1-strip4k",
    [string]$Stage2OutDir = "D:\tmp\mvd-runs\uni-omr-iter13-v2-stage2-24k",
    [string]$EvalOutDir   = "D:\tmp\uni-omr-suite31-ood",
    [string]$TeacherCkpt  = "D:\tmp\mvd-runs\uni-omr-iter10_5-v1\uni_omr_ema_best.pt",
    [string]$Manifest     = "D:\tmp\uni_fullpage_iter13_clean.jsonl",
    [string]$RismStyleDir = "D:\tmp\unlabeled-10k-real\rism_hw",
    [string]$PrintedStrip = "tmp-train/mvd-sysdata-t32",
    [string]$HwStrip      = "D:\tmp\real-hw-strips-iter7",
    [string]$HwStripV06   = "D:\tmp\hw-sim-v06-historical-clean\stage2_strips"
)

$ErrorActionPreference = "Stop"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $RepoRoot

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$TrainScript = "tmp-train/mvd/train_uni_v2.py"
$EvalScript  = "tmp-train/mvd/eval_uni_suite31.py"

if (-not (Test-Path $Python)) {
    throw "Python venv not found at $Python"
}

New-Item -ItemType Directory -Force -Path $Stage1OutDir | Out-Null
New-Item -ItemType Directory -Force -Path $Stage2OutDir | Out-Null
New-Item -ItemType Directory -Force -Path $EvalOutDir   | Out-Null

# ---------------------------------------------------------------------------
# Stage 1: 4,000-step Strip Warmup (eff_batch = 8 x 4 = 32 strips/step)
# ---------------------------------------------------------------------------
$Stage1Args = @(
    "--preset", "small_45m",
    "--curriculum", "strip",
    "--strip-dirs", $PrintedStrip,
    "--hw-strip-dirs", $HwStrip, $HwStripV06,
    "--hw-frac", "0.30",
    "--strip-batch-size", "8",
    "--strip-grad-accum", "4",
    "--grad-accum", "4",
    "--strip-height", "192",
    "--max-strip-width", "1024",
    "--max-seq-len", "1024",
    "--optimizer", "muon",
    "--muon-lr", "1.0e-3",
    "--lr", "3.0e-4",
    "--weight-decay", "0.01",
    "--no-decay-1d",
    "--grad-clip", "1.0",
    "--lr-schedule", "wsd",
    "--warmup-steps", "200",
    "--wsd-decay-frac", "0.15",
    "--min-lr-ratio", "0.05",
    "--steps", "4000",
    "--mixstyle-prob", "0.0",
    "--cass-prob", "0.15",
    "--augment",
    "--aug-prob", "0.75",
    "--teacher-ckpt", $TeacherCkpt,
    "--kd-beta", "0.5",
    "--kd-alpha", "0.30",
    "--kd-temp", "1.5",
    "--kd-teacher-ce-cap", "1.5",
    "--kd-hw-scale", "0.0",
    "--state-token-loss-weight", "2.5",
    "--gpu-ema-decay", "0.995",
    "--gpu-ema-every", "2",
    "--eval-every", "2000",
    "--val-count", "32",
    "--save-every-minutes", "5.0",
    "--bf16",
    "--grad-checkpoint",
    "--no-cudnn",
    "--num-workers", "0",
    "--prefetch-threads", "4",
    "--seed", "42",
    "--out-dir", $Stage1OutDir
)

if (-not $SkipStage1) {
    Write-Host "=== [Iter 13 Stage 1] Launching 4,000-step strip warmup -> $Stage1OutDir ==="
    & $Python -u $TrainScript @Stage1Args
    if ($LASTEXITCODE -ne 0) {
        throw "Iter 13 Stage 1 failed with exit code $LASTEXITCODE"
    }
} else {
    Write-Host "=== [Iter 13 Stage 1] Skipped (-SkipStage1) ==="
}

# Resolve Stage 1 handoff checkpoint (prefer ema_best, fallback to ema_latest)
$Stage1Best   = Join-Path $Stage1OutDir "uni_omr_v2_ema_best.pt"
$Stage1Latest = Join-Path $Stage1OutDir "uni_omr_v2_ema_latest.pt"
$Stage1Ckpt   = if (Test-Path $Stage1Best) { $Stage1Best } elseif (Test-Path $Stage1Latest) { $Stage1Latest } else { $Stage1Best }

# ---------------------------------------------------------------------------
# Stage 2: 24,000-step Pure Fullpage (1408px multi-scale + FDA/AdaIN + QAT-INT4)
# ---------------------------------------------------------------------------
$Stage2Args = @(
    "--preset", "small_45m",
    "--init-from", $Stage1Ckpt,
    "--curriculum", "fullpage",
    "--fullpage-frac", "1.00",
    "--fullpage-manifest", $Manifest,
    "--max-systems-per-page", "16",
    "--stratified-fullpage",
    "--split-archival-hw",
    "--val-count", "16",
    "--val-real-hw-count", "16",
    "--fullpage-max-dim", "1408",
    "--multi-scale",
    "--max-seq-len", "3840",
    "--batch-size", "1",
    "--grad-accum", "8",
    "--optimizer", "muon",
    "--muon-lr", "4.0e-4",
    "--lr", "2.5e-4",
    "--weight-decay", "0.01",
    "--no-decay-1d",
    "--grad-clip", "1.0",
    "--lr-schedule", "wsd",
    "--warmup-steps", "300",
    "--wsd-decay-frac", "0.20",
    "--min-lr-ratio", "0.05",
    "--steps", "24000",
    "--mixstyle-prob", "0.0",
    "--cass-prob", "0.15",
    "--augment",
    "--aug-prob", "0.75",
    "--rism-style-dir", $RismStyleDir,
    "--rism-fda-prob", "0.25",
    "--rism-adain-prob", "0.20",
    "--rism-fda-beta", "0.02",
    "--teacher-ckpt", $TeacherCkpt,
    "--kd-beta", "0.5",
    "--kd-alpha", "0.30",
    "--kd-temp", "1.5",
    "--kd-teacher-ce-cap", "1.5",
    "--kd-hw-scale", "0.0",
    "--state-token-loss-weight", "2.5",
    "--qat-int4",
    "--qat-int4-last-frac", "0.15",
    "--qat-int4-block-size", "64",
    "--gpu-ema-decay", "0.9985",
    "--gpu-ema-every", "5",
    "--eval-every", "4000",
    "--eval-max-len", "1024",
    "--save-every-minutes", "5.0",
    "--bf16",
    "--grad-checkpoint",
    "--no-cudnn",
    "--num-workers", "0",
    "--prefetch-threads", "4",
    "--seed", "42",
    "--out-dir", $Stage2OutDir
)

if (-not $SkipStage2) {
    if (-not (Test-Path $Stage1Ckpt)) {
        throw "Stage 1 handoff checkpoint not found at $Stage1Ckpt"
    }
    Write-Host "=== [Iter 13 Stage 2] Launching 24,000-step pure fullpage -> $Stage2OutDir (init: $Stage1Ckpt) ==="
    & $Python -u $TrainScript @Stage2Args
    if ($LASTEXITCODE -ne 0) {
        throw "Iter 13 Stage 2 failed with exit code $LASTEXITCODE"
    }
} else {
    Write-Host "=== [Iter 13 Stage 2] Skipped (-SkipStage2) ==="
}

# ---------------------------------------------------------------------------
# Post-Stage-2 Automatic suite31_hw Evaluation (45 pieces = 90 pages)
# ---------------------------------------------------------------------------
$Stage2Best   = Join-Path $Stage2OutDir "uni_omr_v2_ema_best.pt"
$Stage2Latest = Join-Path $Stage2OutDir "uni_omr_v2_ema_latest.pt"
$Stage2Ckpt   = if (Test-Path $Stage2Best) { $Stage2Best } elseif (Test-Path $Stage2Latest) { $Stage2Latest } else { $Stage2Best }

$EvalFastOut  = Join-Path $EvalOutDir "iter13_v2_stage2_24k_ema_1408_fsa_mtp.json"
$EvalBeamOut  = Join-Path $EvalOutDir "iter13_v2_stage2_24k_ema_1408_fsa_oemer_mbeam2.json"

$EvalFastArgs = @(
    "--ckpt", $Stage2Ckpt,
    "--max-dim", "1408",
    "--max-len", "3584",
    "--use-grammar",
    "--use-mtp-spec",
    "--bf16",
    "--out", $EvalFastOut
)

$EvalBeamArgs = @(
    "--ckpt", $Stage2Ckpt,
    "--max-dim", "1408",
    "--max-len", "4096",
    "--use-grammar",
    "--use-mtp-spec",
    "--oemer-dewarp",
    "--measure-beam-size", "2",
    "--measure-beam-cap", "4",
    "--measure-beam-lp", "2.0",
    "--measure-beam-max-window", "64",
    "--bf16",
    "--out", $EvalBeamOut
)

if (-not $SkipEval) {
    if (-not (Test-Path $Stage2Ckpt)) {
        throw "Stage 2 checkpoint not found at $Stage2Ckpt"
    }

    Write-Host "=== [Iter 13 Eval Pass 1] Fast promoted suite31_hw (1408px/3584, FSA+MTP-4) ==="
    & $Python -u $EvalScript @EvalFastArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Iter 13 Eval Pass 1 failed with exit code $LASTEXITCODE"
    }

    Write-Host "=== [Iter 13 Eval Pass 2] Promoted high-acc suite31_hw (1408px/4096, FSA+MTP-4+oemer-dewarp+mbeam2_cap4_lp2_w64) ==="
    & $Python -u $EvalScript @EvalBeamArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Iter 13 Eval Pass 2 failed with exit code $LASTEXITCODE"
    }

    Write-Host "=== [Iter 13 Eval Pass 3] Full 125-page OMR-NorthStar (neg-control-gate + fixed mscz-rs export) ==="
    & $Python -u "experiments/2026-11-26T14-40-eval-northstar-iter12-v2/run_eval.py" --ckpt $Stage2Ckpt --system-name "uni-omr-iter13-v2-24k-ema" --out-dir "D:\tmp\northstar-eval-runs\iter13-v2" --device cuda --bf16 --neg-control-gate
    if ($LASTEXITCODE -ne 0) {
        throw "Iter 13 Eval Pass 3 (OMR-NorthStar) failed with exit code $LASTEXITCODE"
    }
} else {
    Write-Host "=== [Iter 13 Eval] Skipped (-SkipEval) ==="
}

Write-Host "=== [Iter 13] Complete! Activate learn-from-suite31, learn-from-northstar, and experiment-finished skills next. ==="
