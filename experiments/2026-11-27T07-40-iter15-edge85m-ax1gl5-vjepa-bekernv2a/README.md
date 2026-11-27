# Iter 15 — `UniOMR v2` (`edge85m` + `ax1_gl5` + Stage-0 V-JEPA+VICReg + `bekern+v2` Phase A + Expanded HW)

- **Experiment ID:** `2026-11-27T07-40-iter15-edge85m-ax1gl5-vjepa-bekernv2a`
- **Status:** `PLANNED`
- **Baseline Commit SHA:** `e426c28` *(+ Iter-15 `edge85m` + `ax1_gl5` + Stage-0 V-JEPA + v2-teacher KD preflight commit)*
- **Architecture:** `UniOMRv2Net` preset **`edge85m`** (`d_model=512`, `~81M` total params, `< 55 MB` WebGPU INT4 ONNX) with **`ax1_gl5`** (`--n-axial-layers 1 --n-global-layers 5`)
- **Platform / GPU:** Local `RTX 4080 SUPER` (`16 GB` VRAM, BF16 native, `$0` cloud cost) **or** Google Colab `H100-80GB` (auto-tuning with `A100-40GB` / `L4-24GB` fallback)
- **Quick Links:**
  - Plan & Decision Tree: [PLAN.md](./PLAN.md)
  - Self-Contained Local 4-Phase Launcher (Stage-0 V-JEPA + Stage-1 Strip + Stage-2 Full-Page + Auto Evals/INT4/ONNX): [launch_iter15.ps1](./launch_iter15.ps1)
  - Recoverable Colab H100/A100/L4 4-Phase Runner (5-min HF checkpoint sync + zero-loss auto-resume): [run_colab_h100.py](./run_colab_h100.py)
  - Stage-0 Pretrainer Script: [pretrain_vjepa_iter15.py](../../tmp-train/mvd/pretrain_vjepa_iter15.py)
  - Outcome *(written on completion)*: `OUTCOME.md`
  - HF Repo Paths (`gregoreesmaa/sheet-music-converter-omr`):
    - Ledger: `runs/2026-11-27T07-40-iter15-edge85m-ax1gl5-vjepa-bekernv2a.json`
    - Checkpoints (local / Colab H100 5-min sync): `checkpoints/2026-11-27T07-40-iter15-edge85m-ax1gl5-vjepa-bekernv2a/` and `checkpoints/iter15-edge85m-h100/`
    - Eval JSONs: `evals/2026-11-27T07-40-iter15-edge85m-ax1gl5-vjepa-bekernv2a_suite31_hw.json`
- **Local Output Dirs:**
  - Phase 0 Stage-0 V-JEPA+VICReg (`3,000` steps): `D:\tmp\mvd-runs\uni-omr-iter15-edge85m-stage0-vjepa`
  - Phase 1 Stage-1 Strip Warmup (`4,000` steps): `D:\tmp\mvd-runs\uni-omr-iter15-edge85m-stage1-strip4k`
  - Phase 2 Stage-2 Full-Page (`20,000` steps): `D:\tmp\mvd-runs\uni-omr-iter15-edge85m-stage2-20k`
  - Phase 3 Eval & WebGPU Export Artifacts:
    - `suite31_hw` Fast (`1408px/3584, FSA+MTP-4, mem=4096`): `D:\tmp\uni-omr-suite31-ood\iter15_edge85m_stage2_20k_ema_latest_1408_fsa_mtp.json`
    - `suite31_hw` `+ILMS (0.40/0.85, mem=4096)`: `D:\tmp\uni-omr-suite31-ood\iter15_edge85m_stage2_20k_ema_latest_1408_fsa_mtp_ilms.json`
    - `suite31_hw` INT4-Dequant Parity Check: `D:\tmp\uni-omr-suite31-ood\iter15_edge85m_stage2_20k_ema_latest_int4dequant_1408_fsa_mtp.json`
    - `OMR-NorthStar` (`125` pages, `ema_latest`, `--neg-control-gate`): `D:\tmp\northstar-eval-runs\iter15-edge85m-last\report.json`
    - WebGPU INT4 ONNX Export (`< 55 MB`): `D:\tmp\mvd-runs\uni-omr-iter15-edge85m-stage2-20k\onnx_webgpu_int4`

---

## Summary & Hypothesis

Iter 15 scales `UniOMR v2` from `small_45m` (`~41.3M`, `ax2_gl4`) to **`edge85m + ax1_gl5`** (`~81M` total params, `1` axial + `5` global 2D-RoPE layers, `< 55 MB` WebGPU INT4) in a clean 4-phase pipeline combining every promoted cloud & forensic win:

1. **Phase 0 (Stage-0 V-JEPA 2.1 + VICReg Encoder Pretraining):** Pretrains the `edge85m + ax1_gl5` backbone + encoder from scratch using our locked `ov-31b` recipe (`mask_ratio=0.45` in `[0.40, 0.50]`, VICReg `var_w=1.0, cov_w=0.04`, which lifted Ridge Staff `R²` `0.210 -> 0.439`) over all **`11,688` unique permissive RISM archival pages** (`5,707` in `D:\tmp\unlabeled-10k-real\rism_hw` + `5,981` extra unique pages in `D:\tmp\rism-pdm-hw\pages`) plus printed pages (`0` eval overlap).
2. **Phase 1 (Stage-1 Strip Warmup, `4,000` steps):** Initializes the `edge85m + ax1_gl5` encoder via `--init-encoder-from` from Phase 0's EMA V-JEPA checkpoint and warms up the `10`-layer GQA decoder + `MTP-4` heads on printed + real/sim handwritten strips (`eff_bs=32`).
3. **Phase 2 (Stage-2 Full-Page Training, `20,000` steps):** Trains full-page `bekern+v2` Phase A (`vocab_size=265`) with **Iter 14 `265`-vocab EMA teacher KD** (`--teacher-ckpt D:\tmp\mvd-runs\uni-omr-iter14-v2a-stage2-16k\uni_omr_v2_ema_latest.pt`), expanded `ov-39` RISM pseudo-labels, newly aligned gold-XML archival HW (`1,466`-page pool), `neg_control` (`--stratified-preset v3`), `--expr-token-loss-weight 2.5`, `--tok-dropout 0.12`, `--adaptive-ink-target`, late-stage INT4 QAT (`--qat-int4 --qat-int4-start-frac 0.85`), and `--save-ema-milestones 4000`.
4. **Phase 3 (Automatic Eval + INT4 Parity + WebGPU `< 55 MB` ONNX Export Check):** Runs `suite31_hw` Fast (`FSA+MTP-4`), `suite31_hw` `+ILMS (0.40/0.85)`, `suite31_hw` INT4-dequant parity check (verifying `< 0.5 pp` drop under INT4 quantization), 125-page `OMR-NorthStar` (`--neg-control-gate`), and WebGPU INT4 ONNX export (`< 55 MB` size gate).

**Hypothesis:** Combining `2×` model capacity (`edge85m`) + global-heavy `ax1_gl5` encoder + `11,688`-page Stage-0 V-JEPA+VICReg archival pretraining + Iter 14 `265`-vocab EMA teacher KD + expanded gold-XML archival HW (`1,466` pages) will substantially lift handwritten & full-page accuracy (`suite31_hw` `all45_clean >= 0.740`, `hw_clean >= 0.340`, `OMR-NorthStar` `NS >= 45.0`, `HWM >= 34.0`) while staying `< 55 MB` in WebGPU INT4 with `< 0.5 pp` INT4-dequant parity loss.

---

## Baselines to Beat

| Checkpoint | Benchmark / Arm | Headline / Key Metrics |
|---|---|---|
| **Iter 13 `v2` (`small_45m`, `ema_best`)** | `suite31_hw` Fast (`1408/3584, FSA+MTP-4`) | `pr_clean=0.9193`, `pr_photo=0.9103`, `hw_clean=0.2448`, `hw_photo=0.2533`, **`all45_clean=0.7094`**, **`all45_photo=0.7059`** |
| **Iter 13 `v2` (`small_45m`, `ema_latest`)** | `OMR-NorthStar` (`125p, --neg-control-gate`) | **`NS=34.08`**, `Tier-W NS=36.45`, `CDR=82.94`, `ALS=78.47`, `RWP=80.93`, `HWM=23.31`, `S_expr=0.5544`, `S_struct=0.7905`, `NMHR=0.0%` |
| **Iter 14 `v2a` (`small_45m`, `ema_latest`)** | `suite31_hw` + `OMR-NorthStar` | *(Parent teacher baseline — record final measured numbers once Iter 14 completes)* |
| **Iter 15 `edge85m` Target (`PROMOTE`)** | `suite31_hw` + `OMR-NorthStar` + INT4 + WebGPU | `all45_clean >= 0.740`, `all45_photo >= 0.730`, `NS >= 45.0`, `HWM >= 34.0`, `NMHR == 0.0%`, INT4 parity `<= 0.5 pp`, WebGPU `< 55 MB` |

---

## Colab H100 Recoverable Launch (Auto-Tunes to H100 / A100 / L4)

`run_colab_h100.py` runs all 4 phases with automatic GPU tier detection (`H100 >= 70 GB`: `bs=8, ga=1`, no grad checkpoint; `A100 >= 36 GB`: `bs=4, ga=2`; `L4 < 36 GB`: `bs=2, ga=4`), `BF16` + cuDNN NCHW + `--fixed-hw-buckets` + `BatchedMuon`, a 5-minute background HF checkpoint syncer (`checkpoints/iter15-edge85m-h100/`), and zero-loss auto-resume from HF `phase_state.json` on reconnect.

```powershell
# 1. CPU dry-run self-test (GPU auto-tuner, D:\tmp -> /content/data remapper, 5-min HF syncer)
$env:CUDA_VISIBLE_DEVICES = ""
.\.venv\Scripts\python.exe experiments/2026-11-27T07-40-iter15-edge85m-ax1gl5-vjepa-bekernv2a/run_colab_h100.py --dry-run

# 2. Launch on Colab H100 (named persistent session with clean teardown in finally)
.\.venv\Scripts\colab.exe new -s iter15-h100 --gpu H100
try {
    .\.venv\Scripts\colab.exe upload -s iter15-h100 "$env:USERPROFILE\.cache\huggingface\token" /tmp/.hf_token
    .\.venv\Scripts\colab.exe upload -s iter15-h100 iter15_bundle.tar.gz /content/iter15_bundle.tar.gz
    .\.venv\Scripts\colab.exe exec -s iter15-h100 --timeout 0 -f experiments/2026-11-27T07-40-iter15-edge85m-ax1gl5-vjepa-bekernv2a/run_colab_h100.py
} finally {
    .\.venv\Scripts\colab.exe stop -s iter15-h100
}
```
