<#
.SYNOPSIS
    Canonical 3-phase Iter 14 UniOMR v2 (bekern+v2 Phase A + ov-39 RISM pseudo-labels)
    launcher + automatic suite31_hw (Fast & +ILMS) and 125-page OMR-NorthStar evaluation.

.NOTES
    - PowerShell-safe (no `&&`, strict `$ErrorActionPreference = "Stop"`).
    - Local RTX 4080 SUPER (16 GB VRAM, BF16 native, CUTLASS EFFICIENT_ATTENTION,
      cuDNN NCHW with --fixed-hw-buckets).
    - Warm-starts from Iter 13 uni_omr_v2_ema_latest.pt (vocab 237 -> 265 append-only
      with exact PAD/BOS/EOS/SYS remapping 233..236 -> 261..264).
    - Pass `-SkipPhase0`, `-SkipTrain`, `-SkipEval`, or `-NoCudnn` to resume cleanly.
#>
[CmdletBinding()]
param(
    [switch]$SkipPhase0,
    [switch]$SkipTrain,
    [switch]$SkipEval,
    [switch]$NoCudnn,
    [string]$InitCkpt        = "D:\tmp\mvd-runs\uni-omr-iter13-v2-stage2-24k\uni_omr_v2_ema_latest.pt",
    [string]$TeacherCkpt     = "D:\tmp\mvd-runs\uni-omr-iter10_5-v1\uni_omr_ema_best.pt",
    [string]$RismStyleDir    = "D:\tmp\unlabeled-10k-real\rism_hw",
    [string]$RismPseudoOut   = "D:\tmp\rism-pseudolabels-ov39\uni_rism_pseudolabels_ov39.jsonl",
    [int]$RismPseudoLimit    = 1200,
    [string]$Manifest        = "D:\tmp\uni_fullpage_iter14_v2a_rism.jsonl",
    [string]$Stage2OutDir    = "D:\tmp\mvd-runs\uni-omr-iter14-v2a-stage2-16k",
    [string]$EvalOutDir      = "D:\tmp\uni-omr-suite31-ood",
    [string]$NorthStarOutDir = "D:\tmp\northstar-eval-runs\iter14-v2a-last",
    [int]$Stage2Steps        = 16000,
    [int]$Stage2Warmup       = 300
)

$ErrorActionPreference = "Stop"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $RepoRoot

$Python           = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$PseudoScript     = "tmp-train/mvd/build_rism_pseudolabel_manifest.py"
$ManifestScript   = "tmp-train/mvd/build_iter14_manifest.py"
$TrainScript      = "tmp-train/mvd/train_uni_v2.py"
$EvalScript       = "tmp-train/mvd/eval_uni_suite31.py"
$NorthStarScript  = "experiments/2026-11-26T14-40-eval-northstar-iter12-v2/run_eval.py"
$AssemblyBin      = Join-Path $RepoRoot "converter\assembly-serialize\target\release\assembly-serialize.exe"

if (-not (Test-Path $Python)) {
    throw "Python venv not found at $Python"
}
if (-not (Test-Path $AssemblyBin)) {
    throw "Rebuilt assembly-serialize.exe not found at $AssemblyBin"
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $RismPseudoOut) | Out-Null
New-Item -ItemType Directory -Force -Path $Stage2OutDir | Out-Null
New-Item -ItemType Directory -Force -Path $EvalOutDir   | Out-Null
New-Item -ItemType Directory -Force -Path $NorthStarOutDir | Out-Null

# ---------------------------------------------------------------------------
# Phase 0: Generate ov-39 RISM Pseudo-Labels + Rebuild Iter 14 v2a+RISM Manifest
# ---------------------------------------------------------------------------
$PseudoArgs = @(
    "--ckpt", $InitCkpt,
    "--input-dir", $RismStyleDir,
    "--out", $RismPseudoOut,
    "--device", "cuda",
    "--bf16",
    "--max-dim", "1408",
    "--limit", "$RismPseudoLimit",
    "--resume"
)

$ManifestArgs = @(
    "--reexport-v2",
    "--rism-pseudo-jsonl", $RismPseudoOut,
    "--out", $Manifest
)

if (-not $SkipPhase0) {
    if (-not (Test-Path $InitCkpt)) {
        throw "Init/pseudo-label checkpoint not found at $InitCkpt"
    }
    Write-Host "=== [Iter 14 Phase 0a] Generating ov-39 RISM pseudo-labels (limit=$RismPseudoLimit, 1408px, bf16) -> $RismPseudoOut ==="
    & $Python -u $PseudoScript @PseudoArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Iter 14 Phase 0a (ov-39 RISM pseudo-labels) failed with exit code $LASTEXITCODE"
    }

    Write-Host "=== [Iter 14 Phase 0b] Rebuilding bekern+v2 Phase A + RISM manifest -> $Manifest ==="
    & $Python -u $ManifestScript @ManifestArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Iter 14 Phase 0b (build_iter14_manifest) failed with exit code $LASTEXITCODE"
    }
} else {
    Write-Host "=== [Iter 14 Phase 0] Skipped (-SkipPhase0) ==="
}

# ---------------------------------------------------------------------------
# Phase 1: Iter 14 Stage 2 (16,000 steps, bekern+v2a + v3 strata + Tier 1-3 speedups)
# ---------------------------------------------------------------------------
$Stage2Args = @(
    "--preset", "small_45m",
    "--max-mem-tokens", "3072",
    "--init-from", $InitCkpt,
    "--curriculum", "fullpage",
    "--fullpage-frac", "1.00",
    "--fullpage-manifest", $Manifest,
    "--max-systems-per-page", "16",
    "--stratified-fullpage",
    "--stratified-preset", "v3",
    "--val-count", "16",
    "--val-real-hw-count", "16",
    "--fullpage-max-dim", "1408",
    "--multi-scale",
    "--fixed-hw-buckets",
    "--max-seq-len", "3840",
    "--batch-size", "2",
    "--grad-accum", "4",
    "--optimizer", "muon",
    "--muon-lr", "4.0e-4",
    "--lr", "2.5e-4",
    "--weight-decay", "0.01",
    "--no-decay-1d",
    "--grad-clip", "1.0",
    "--lr-schedule", "wsd",
    "--warmup-steps", "$Stage2Warmup",
    "--wsd-decay-frac", "0.20",
    "--min-lr-ratio", "0.05",
    "--steps", "$Stage2Steps",
    "--mixstyle-prob", "0.0",
    "--cass-batch-prob", "0.5",
    "--cass-prob", "0.30",
    "--tok-dropout", "0.12",
    "--adaptive-ink-target",
    "--augment",
    "--aug-prob", "0.75",
    "--rism-style-dir", $RismStyleDir,
    "--rism-fda-prob", "0.25",
    "--rism-adain-prob", "0.20",
    "--rism-fda-beta", "0.02",
    "--teacher-ckpt", $TeacherCkpt,
    "--teacher-max-dim", "1024",
    "--kd-beta", "0.5",
    "--kd-alpha", "0.30",
    "--kd-temp", "1.5",
    "--kd-teacher-ce-cap", "1.5",
    "--kd-hw-scale", "0.0",
    "--kd-prob", "0.5",
    "--kd-stop-frac", "0.65",
    "--state-token-loss-weight", "2.5",
    "--expr-token-loss-weight", "2.5",
    "--qat-int4",
    "--qat-int4-last-frac", "0.15",
    "--qat-int4-block-size", "64",
    "--gpu-ema-decay", "0.9985",
    "--gpu-ema-every", "5",
    "--save-ema-milestones", "4000",
    "--best-min-step", "4000",
    "--eval-every", "4000",
    "--eval-max-len", "1024",
    "--save-every-minutes", "5.0",
    "--bf16",
    "--grad-checkpoint",
    "--num-workers", "0",
    "--prefetch-threads", "6",
    "--seed", "42",
    "--out-dir", $Stage2OutDir
)
if ($NoCudnn) {
    $Stage2Args += "--no-cudnn"
}

if (-not $SkipTrain) {
    if (-not (Test-Path $InitCkpt)) {
        throw "Warm-start checkpoint not found at $InitCkpt"
    }
    if (-not (Test-Path $Manifest)) {
        throw "Iter 14 full-page manifest not found at $Manifest"
    }
    Write-Host "=== [Iter 14 Phase 1] Launching $Stage2Steps-step bekern+v2a Stage 2 -> $Stage2OutDir (init: $InitCkpt) ==="
    & $Python -u $TrainScript @Stage2Args
    if ($LASTEXITCODE -ne 0) {
        throw "Iter 14 Phase 1 (Stage 2 training) failed with exit code $LASTEXITCODE"
    }
} else {
    Write-Host "=== [Iter 14 Phase 1] Skipped (-SkipTrain) ==="
}

# ---------------------------------------------------------------------------
# Phase 2: Automatic Post-Training Evals (prefer post-WSD ema_latest)
# ---------------------------------------------------------------------------
$Stage2Latest = Join-Path $Stage2OutDir "uni_omr_v2_ema_latest.pt"
$Stage2Best   = Join-Path $Stage2OutDir "uni_omr_v2_ema_best.pt"
$EvalCkpt     = if (Test-Path $Stage2Latest) { $Stage2Latest } elseif (Test-Path $Stage2Best) { $Stage2Best } else { $Stage2Latest }

$EvalFastOut  = Join-Path $EvalOutDir "iter14_v2a_stage2_16k_ema_latest_1408_fsa_mtp.json"
$EvalIlmsOut  = Join-Path $EvalOutDir "iter14_v2a_stage2_16k_ema_latest_1408_fsa_mtp_ilms.json"

$EvalFastArgs = @(
    "--ckpt", $EvalCkpt,
    "--max-dim", "1408",
    "--max-len", "3584",
    "--max-mem-tokens", "4096",
    "--use-grammar",
    "--use-mtp-spec",
    "--bf16",
    "--out", $EvalFastOut
)

$EvalIlmsArgs = @(
    "--ckpt", $EvalCkpt,
    "--max-dim", "1408",
    "--max-len", "3584",
    "--max-mem-tokens", "4096",
    "--use-grammar",
    "--use-mtp-spec",
    "--ilms-alpha", "0.40",
    "--ilms-conf-thr", "0.85",
    "--bf16",
    "--out", $EvalIlmsOut
)

$NorthStarArgs = @(
    "--ckpt", $EvalCkpt,
    "--aser-bin", $AssemblyBin,
    "--system-name", "uni-omr-iter14-v2a-16k-ema-last",
    "--out-dir", $NorthStarOutDir,
    "--device", "cuda",
    "--bf16",
    "--neg-control-gate"
)

if (-not $SkipEval) {
    if (-not (Test-Path $EvalCkpt)) {
        throw "Iter 14 checkpoint not found at $EvalCkpt"
    }

    Write-Host "=== [Iter 14 Eval Pass 1] suite31_hw Fast (1408px/3584, FSA+MTP-4, max_mem_tokens=4096) ==="
    & $Python -u $EvalScript @EvalFastArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Iter 14 Eval Pass 1 (suite31_hw Fast) failed with exit code $LASTEXITCODE"
    }

    Write-Host "=== [Iter 14 Eval Pass 2] suite31_hw +ILMS (0.40/0.85, max_mem_tokens=4096) ==="
    & $Python -u $EvalScript @EvalIlmsArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Iter 14 Eval Pass 2 (suite31_hw +ILMS) failed with exit code $LASTEXITCODE"
    }

    Write-Host "=== [Iter 14 Eval Pass 3] Full 125-page OMR-NorthStar on ema_latest (--neg-control-gate + rebuilt assembly-serialize.exe) ==="
    & $Python -u $NorthStarScript @NorthStarArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Iter 14 Eval Pass 3 (OMR-NorthStar) failed with exit code $LASTEXITCODE"
    }
} else {
    Write-Host "=== [Iter 14 Phase 2] Skipped (-SkipEval) ==="
}

Write-Host "=== [Iter 14] Complete! Activate learn-from-suite31, learn-from-northstar, and experiment-finished skills next. ==="
