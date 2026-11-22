<#
.SYNOPSIS
Launch supervised MvdNet training with the verified RTX 4080 SUPER defaults
(B=6, accum=1, bf16, channels_last, width-bucket, prefetch=4, and 16-stage
physical + 3D off-center geometric per-epoch augmentation enabled by default).

.EXAMPLE
.\scripts\run-train.ps1 --renders D:\tmp\corpus10x\pdmx\unified-sysdata --targets D:\tmp\corpus10x\pdmx\unified-sysdata --out D:\tmp\mvd-runs\routeA --steps 10000 --pretrained --stride 4x8 --extra-chans 8 --hybrid-cache D:\tmp\corpus10x\pdmx\unified-hybrid\hybrid_staff.jsonl
#>
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$TrainArgs
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Py = Join-Path $RepoRoot '.venv\Scripts\python.exe'
$Sup = Join-Path $RepoRoot 'tmp-train\mvd\train_supervise.py'

if (-not (Test-Path $Py)) {
    throw "Virtualenv Python not found at $Py"
}

$env:PYTORCH_CUDA_ALLOC_CONF = 'expandable_segments:True'
& $Py $Sup --tries 20 --cooldown 60 -- @TrainArgs
exit $LASTEXITCODE
