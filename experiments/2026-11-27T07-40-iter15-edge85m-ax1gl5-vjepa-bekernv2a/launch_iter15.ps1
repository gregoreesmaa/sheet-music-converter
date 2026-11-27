<#
.SYNOPSIS
    Canonical 4-phase Iter 15 UniOMR v2 (edge85m + ax1_gl5 + Stage-0 V-JEPA+VICReg
    + bekern+v2 Phase A + Iter 14 265-vocab EMA Teacher KD + Late INT4 QAT) launcher
    with automatic suite31_hw (Fast, +ILMS, INT4-dequant parity), 125-page
    OMR-NorthStar, and WebGPU INT4 (< 55 MB) ONNX export verification.

.NOTES
    - PowerShell-safe (no `&&`, strict `$ErrorActionPreference = "Stop"`).
    - Local RTX 4080 SUPER (16 GB VRAM, BF16 native, CUTLASS EFFICIENT_ATTENTION).
    - Pass `-SkipPhase0`, `-SkipPhase1`, `-SkipPhase2`, `-SkipPhase3`, or `-NoCudnn`
      to resume cleanly from any phase.
#>
[CmdletBinding()]
param(
    [switch]$SkipPhase0,
    [switch]$SkipPhase1,
    [switch]$SkipPhase2,
    [switch]$SkipPhase3,
    [switch]$NoCudnn,
    [string]$RismDirA          = "D:\tmp\unlabeled-10k-real\rism_hw",
    [string]$RismDirB          = "D:\tmp\rism-pdm-hw\pages",
    [string]$PrintedManifest   = "D:\tmp\uni_fullpage_iter14_v2a.jsonl",
    [int]$PrintedMaxPages      = 6000,
    [string]$PrintedStrip      = "tmp-train/mvd-sysdata-t32",
    [string]$HwStrip           = "D:\tmp\real-hw-strips-iter7",
    [string]$HwStripV06        = "D:\tmp\hw-sim-v06-historical-clean\stage2_strips",
    [string]$Stage2Manifest    = "D:\tmp\uni_fullpage_iter15_v2a.jsonl",
    [string]$Iter14TeacherCkpt = "D:\tmp\mvd-runs\uni-omr-iter14-v2a-stage2-16k\uni_omr_v2_ema_latest.pt",
    [string]$Stage0OutDir      = "D:\tmp\mvd-runs\uni-omr-iter15-edge85m-stage0-vjepa",
    [string]$Stage1OutDir      = "D:\tmp\mvd-runs\uni-omr-iter15-edge85m-stage1-strip4k",
    [string]$Stage2OutDir      = "D:\tmp\mvd-runs\uni-omr-iter15-edge85m-stage2-20k",
    [string]$EvalOutDir        = "D:\tmp\uni-omr-suite31-ood",
    [string]$NorthStarOutDir   = "D:\tmp\northstar-eval-runs\iter15-edge85m-last",
    [string]$Stage1InitEnc     = "",
    [string]$Stage2InitFrom    = "",
    [int]$Stage0Steps          = 3000,
    [int]$Stage1Steps          = 4000,
    [int]$Stage2Steps          = 20000,
    [int]$Stage2Warmup         = 300,
    [double]$WebGpuMaxMb       = 55.0
)

$ErrorActionPreference = "Stop"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $RepoRoot

$Python          = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$VjepaScript     = "tmp-train/mvd/pretrain_vjepa_iter15.py"
$TrainScript     = "tmp-train/mvd/train_uni_v2.py"
$EvalScript      = "tmp-train/mvd/eval_uni_suite31.py"
$OnnxScript      = "tmp-train/mvd/export_uni_v2_onnx.py"
$NorthStarScript = "experiments/2026-11-26T14-40-eval-northstar-iter12-v2/run_eval.py"
$AssemblyBin     = Join-Path $RepoRoot "converter\assembly-serialize\target\release\assembly-serialize.exe"

if (-not (Test-Path $Python)) {
    throw "Python venv not found at $Python"
}
if (-not (Test-Path $AssemblyBin)) {
    throw "Rebuilt assembly-serialize.exe not found at $AssemblyBin"
}

New-Item -ItemType Directory -Force -Path $Stage0OutDir    | Out-Null
New-Item -ItemType Directory -Force -Path $Stage1OutDir    | Out-Null
New-Item -ItemType Directory -Force -Path $Stage2OutDir    | Out-Null
New-Item -ItemType Directory -Force -Path $EvalOutDir      | Out-Null
New-Item -ItemType Directory -Force -Path $NorthStarOutDir | Out-Null

# ---------------------------------------------------------------------------
# Phase 0: Stage-0 V-JEPA 2.1 + VICReg Pretraining (edge85m + ax1_gl5)
# ---------------------------------------------------------------------------
$Stage0Args = @(
    "--arch-preset", "edge85m",
    "--n-axial-layers", "1",
    "--n-global-layers", "5",
    "--max-mem-tokens", "3072",
    "--adaptive-ink-target",
    "--image-dirs", $RismDirA, $RismDirB,
    "--printed-manifest", $PrintedManifest,
    "--printed-max-pages", "$PrintedMaxPages",
    "--mask-ratio", "0.45",
    "--mask-half-width", "0.05",
    "--vicreg-var-weight", "1.0",
    "--vicreg-cov-weight", "0.04",
    "--disable-random-gram",
    "--stroke-hog-weight", "0.25",
    "--router-cons-weight", "0.2",
    "--optimizer", "muon",
    "--lr", "3.0e-4",
    "--weight-decay", "0.04",
    "--warmup-steps", "150",
    "--steps", "$Stage0Steps",
    "--batch-size", "2",
    "--grad-accum", "4",
    "--max-dim", "1024",
    "--bf16",
    "--grad-checkpoint",
    "--no-cudnn",
    "--prefetch-threads", "4",
    "--val-probe-pages", "128",
    "--save-every", "500",
    "--save-every-minutes", "5.0",
    "--seed", "42",
    "--out-dir", $Stage0OutDir
)

if (-not $SkipPhase0) {
    Write-Host "=== [Iter 15 Phase 0] Stage-0 V-JEPA+VICReg ($Stage0Steps steps, edge85m + ax1_gl5) -> $Stage0OutDir ==="
    & $Python -u $VjepaScript @Stage0Args
    if ($LASTEXITCODE -ne 0) {
        throw "Iter 15 Phase 0 (Stage-0 V-JEPA+VICReg) failed with exit code $LASTEXITCODE"
    }
} else {
    Write-Host "=== [Iter 15 Phase 0] Skipped (-SkipPhase0) ==="
}

$Stage0Last = Join-Path $Stage0OutDir "uni_omr_v2_vjepa_last.pt"
$Stage0Best = Join-Path $Stage0OutDir "uni_omr_v2_vjepa_best.pt"
$Phase0Ckpt = if ($Stage1InitEnc) { $Stage1InitEnc } elseif (Test-Path $Stage0Last) { $Stage0Last } else { $Stage0Best }

# ---------------------------------------------------------------------------
# Phase 1: Stage-1 Strip Warmup (4,000 steps, edge85m + ax1_gl5)
# ---------------------------------------------------------------------------
$Stage1Args = @(
    "--preset", "edge85m",
    "--n-axial-layers", "1",
    "--n-global-layers", "5",
    "--max-mem-tokens", "3072",
    "--adaptive-ink-target",
    "--init-encoder-from", $Phase0Ckpt,
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
    "--muon-lr", "8.0e-4",
    "--lr", "2.5e-4",
    "--weight-decay", "0.01",
    "--no-decay-1d",
    "--grad-clip", "1.0",
    "--lr-schedule", "wsd",
    "--warmup-steps", "200",
    "--wsd-decay-frac", "0.15",
    "--min-lr-ratio", "0.05",
    "--steps", "$Stage1Steps",
    "--mixstyle-prob", "0.0",
    "--cass-prob", "0.15",
    "--tok-dropout", "0.12",
    "--augment",
    "--aug-prob", "0.75",
    "--state-token-loss-weight", "2.5",
    "--expr-token-loss-weight", "2.5",
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

if (-not $SkipPhase1) {
    if (-not (Test-Path $Phase0Ckpt)) {
        throw "Phase 0 V-JEPA encoder checkpoint not found at $Phase0Ckpt"
    }
    Write-Host "=== [Iter 15 Phase 1] Stage-1 Strip Warmup ($Stage1Steps steps, edge85m + ax1_gl5) -> $Stage1OutDir ==="
    & $Python -u $TrainScript @Stage1Args
    if ($LASTEXITCODE -ne 0) {
        throw "Iter 15 Phase 1 (Stage-1 Strip Warmup) failed with exit code $LASTEXITCODE"
    }
} else {
    Write-Host "=== [Iter 15 Phase 1] Skipped (-SkipPhase1) ==="
}

$Stage1Best   = Join-Path $Stage1OutDir "uni_omr_v2_ema_best.pt"
$Stage1Latest = Join-Path $Stage1OutDir "uni_omr_v2_ema_latest.pt"
$Phase1Ckpt   = if ($Stage2InitFrom) { $Stage2InitFrom } elseif (Test-Path $Stage1Best) { $Stage1Best } else { $Stage1Latest }

# ---------------------------------------------------------------------------
# Phase 2: Stage-2 Full-Page (20,000 steps, bekern+v2a + Iter 14 KD + Late INT4 QAT)
# ---------------------------------------------------------------------------
$Stage2Args = @(
    "--preset", "edge85m",
    "--n-axial-layers", "1",
    "--n-global-layers", "5",
    "--max-mem-tokens", "3072",
    "--adaptive-ink-target",
    "--init-from", $Phase1Ckpt,
    "--curriculum", "fullpage",
    "--fullpage-frac", "1.00",
    "--fullpage-manifest", $Stage2Manifest,
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
    "--muon-lr", "3.5e-4",
    "--lr", "2.0e-4",
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
    "--augment",
    "--aug-prob", "0.75",
    "--rism-style-dir", $RismDirA,
    "--rism-fda-prob", "0.25",
    "--rism-adain-prob", "0.20",
    "--rism-fda-beta", "0.02",
    "--teacher-ckpt", $Iter14TeacherCkpt,
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
    "--qat-int4-start-frac", "0.85",
    "--qat-int4-block-size", "32",
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

if (-not $SkipPhase2) {
    if (-not (Test-Path $Phase1Ckpt)) {
        throw "Phase 1 handoff checkpoint not found at $Phase1Ckpt"
    }
    if (-not (Test-Path $Stage2Manifest)) {
        throw "Stage-2 full-page manifest not found at $Stage2Manifest"
    }
    if (-not (Test-Path $Iter14TeacherCkpt)) {
        throw "Iter 14 265-vocab EMA teacher checkpoint not found at $Iter14TeacherCkpt"
    }
    Write-Host "=== [Iter 15 Phase 2] Stage-2 Full-Page ($Stage2Steps steps, edge85m + ax1_gl5 + bekern+v2a + Iter 14 KD + Late INT4 QAT) -> $Stage2OutDir ==="
    & $Python -u $TrainScript @Stage2Args
    if ($LASTEXITCODE -ne 0) {
        throw "Iter 15 Phase 2 (Stage-2 Full-Page) failed with exit code $LASTEXITCODE"
    }
} else {
    Write-Host "=== [Iter 15 Phase 2] Skipped (-SkipPhase2) ==="
}

# ---------------------------------------------------------------------------
# Phase 3: Auto Evals (suite31_hw Fast, +ILMS, INT4 Parity, OMR-NorthStar, WebGPU ONNX)
# ---------------------------------------------------------------------------
$Stage2Latest = Join-Path $Stage2OutDir "uni_omr_v2_ema_latest.pt"
$Stage2Best   = Join-Path $Stage2OutDir "uni_omr_v2_ema_best.pt"
$EvalCkpt     = if (Test-Path $Stage2Latest) { $Stage2Latest } elseif (Test-Path $Stage2Best) { $Stage2Best } else { $Stage2Latest }

$EvalFastOut      = Join-Path $EvalOutDir "iter15_edge85m_stage2_20k_ema_latest_1408_fsa_mtp.json"
$EvalIlmsOut      = Join-Path $EvalOutDir "iter15_edge85m_stage2_20k_ema_latest_1408_fsa_mtp_ilms.json"
$Int4DequantCkpt  = Join-Path $Stage2OutDir "uni_omr_v2_ema_latest_int4dequant.pt"
$EvalInt4Out      = Join-Path $EvalOutDir "iter15_edge85m_stage2_20k_ema_latest_int4dequant_1408_fsa_mtp.json"
$OnnxOutDir       = Join-Path $Stage2OutDir "onnx_webgpu_int4"

if (-not $SkipPhase3) {
    if (-not (Test-Path $EvalCkpt)) {
        throw "Iter 15 checkpoint not found at $EvalCkpt"
    }

    Write-Host "=== [Iter 15 Phase 3.1] suite31_hw Fast (1408px/3584, FSA+MTP-4, max_mem_tokens=4096) ==="
    & $Python -u $EvalScript --ckpt $EvalCkpt --max-dim 1408 --max-len 3584 --max-mem-tokens 4096 --use-grammar --use-mtp-spec --bf16 --out $EvalFastOut
    if ($LASTEXITCODE -ne 0) {
        throw "Iter 15 Phase 3.1 (suite31_hw Fast) failed with exit code $LASTEXITCODE"
    }

    Write-Host "=== [Iter 15 Phase 3.2] suite31_hw +ILMS (0.40/0.85, max_mem_tokens=4096) ==="
    & $Python -u $EvalScript --ckpt $EvalCkpt --max-dim 1408 --max-len 3584 --max-mem-tokens 4096 --use-grammar --use-mtp-spec --ilms-alpha 0.40 --ilms-conf-thr 0.85 --bf16 --out $EvalIlmsOut
    if ($LASTEXITCODE -ne 0) {
        throw "Iter 15 Phase 3.2 (suite31_hw +ILMS) failed with exit code $LASTEXITCODE"
    }

    Write-Host "=== [Iter 15 Phase 3.3] Materializing INT4-dequantized checkpoint & running suite31_hw INT4 parity check ==="
    $Int4PrepPy = @"
import sys, torch
sys.path.insert(0, 'tmp-train/mvd')
from uni_omr_v2 import UniOMRv2Config, UniOMRv2Net, QATInt4ControllerV2, fake_quant_int4_ste
ck = torch.load(r'$EvalCkpt', map_location='cpu', weights_only=True)
cfg = UniOMRv2Config(**ck['cfg'])
m = UniOMRv2Net(cfg)
m.load_state_dict(ck['model'], strict=False)
ctrl = QATInt4ControllerV2(m, block_size=32, enabled=True)
with torch.no_grad():
    for mod in ctrl._targets:
        mod.weight.copy_(fake_quant_int4_ste(mod.weight, block_size=32))
ctrl.detach()
ck['model'] = {k: v.detach().cpu() for k, v in m.state_dict().items()}
torch.save(ck, r'$Int4DequantCkpt')
print(f'[INT4-dequant] quantized {ctrl.num_layers} Linear layers -> $Int4DequantCkpt')
"@
    & $Python -u -c $Int4PrepPy
    if ($LASTEXITCODE -ne 0) {
        throw "Iter 15 Phase 3.3a (INT4-dequant ckpt prep) failed with exit code $LASTEXITCODE"
    }

    & $Python -u $EvalScript --ckpt $Int4DequantCkpt --max-dim 1408 --max-len 3584 --max-mem-tokens 4096 --use-grammar --use-mtp-spec --bf16 --out $EvalInt4Out
    if ($LASTEXITCODE -ne 0) {
        throw "Iter 15 Phase 3.3b (suite31_hw INT4-dequant parity) failed with exit code $LASTEXITCODE"
    }

    Write-Host "=== [Iter 15 Phase 3.4] Full 125-page OMR-NorthStar on ema_latest (--neg-control-gate + rebuilt assembly-serialize.exe) ==="
    & $Python -u $NorthStarScript --ckpt $EvalCkpt --aser-bin $AssemblyBin --system-name "uni-omr-iter15-edge85m-20k-ema-last" --out-dir $NorthStarOutDir --device cuda --bf16 --neg-control-gate
    if ($LASTEXITCODE -ne 0) {
        throw "Iter 15 Phase 3.4 (OMR-NorthStar) failed with exit code $LASTEXITCODE"
    }

    Write-Host "=== [Iter 15 Phase 3.5] WebGPU INT4 ONNX Export & < $WebGpuMaxMb MB Size Gate -> $OnnxOutDir ==="
    & $Python -u $OnnxScript --ckpt $EvalCkpt --preset edge85m --targets webgpu --fp16 --nbits-block-size 32 --out-dir $OnnxOutDir
    if ($LASTEXITCODE -ne 0) {
        throw "Iter 15 Phase 3.5 (WebGPU INT4 ONNX export) failed with exit code $LASTEXITCODE"
    }

    $WebGpuDir = Join-Path $OnnxOutDir "webgpu"
    $TotalBytes = (Get-ChildItem -Path $WebGpuDir -File -Recurse | Measure-Object -Property Length -Sum).Sum
    $TotalMb = [math]::Round(($TotalBytes / 1MB), 2)
    Write-Host "=== [Iter 15 Phase 3.5] WebGPU INT4 total artifact size: $TotalMb MB (gate: < $WebGpuMaxMb MB) ==="
    if ($TotalMb -ge $WebGpuMaxMb) {
        throw "WebGPU INT4 ONNX export exceeded $WebGpuMaxMb MB gate: $TotalMb MB"
    }
} else {
    Write-Host "=== [Iter 15 Phase 3] Skipped (-SkipPhase3) ==="
}

Write-Host "=== [Iter 15] Complete! Activate learn-from-suite31, learn-from-northstar, and experiment-finished skills next. ==="
