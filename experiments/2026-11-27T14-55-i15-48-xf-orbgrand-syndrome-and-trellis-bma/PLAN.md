# PLAN — `2026-11-27T14-55-i15-48-xf-orbgrand-syndrome-and-trellis-bma`

## 1. Platform & Budget
- **Platform:** Google Colab `1× NVIDIA A100` (or Local `RTX 4080 SUPER` 16 GB, native BF16).
- **Estimated Wall-Clock:** ~12 min total (zero training; 12 hard `suite31_hw` pages × 3 inference arms).
- **Baseline Commit SHA:** `6055c49`.
- **Self-Containment:** All ORBGRAND, SOGRAND, and Trellis-BMA decoding code lives strictly inside `run_colab.py`.

## 2. Identifiers for Follow-Up Queries
- **Experiment ID:** `2026-11-27T14-55-i15-48-xf-orbgrand-syndrome-and-trellis-bma`
- **Hugging Face (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-27T14-55-i15-48-xf-orbgrand-syndrome-and-trellis-bma.json`
  - Run Ledger: `runs/2026-11-27T14-55-i15-48-xf-orbgrand-syndrome-and-trellis-bma.json`
  - Partial Arm Sync: `checkpoints/2026-11-27T14-55-i15-48-xf-orbgrand-syndrome-and-trellis-bma/{greedy_fsa,orbgrand_sogrand,trellis_bma_3view}_arm_result.json`

## 3. Data, Checkpoint & 3-Arm Inference Protocol
- **Checkpoint Resolution (auto-fallback):** Checks `UNIOMR_CKPT` env → HF `checkpoints/iter14-final/uni_omr_v2_ema_latest.pt` → `checkpoints/iter14-v2a-mid/uni_omr_v2_ema_latest.pt` → `checkpoints/iter10_soup_baseline/uni_omr_soup.pt`.
- **Eval Subset:** Deterministic 12-page hard printed + HW subset of `datasets/suite31_hw_eval.tar` (6 hard printed `photo` pages + 6 hard real handwritten `photo`/`clean` pages; permissive/benchmark provenance, zero `CC-BY-NC`).
- **Windows RTX 4080 SUPER Safety:** Automatically disables `torch.backends.cudnn.enabled` and forces CUTLASS `EFFICIENT_ATTENTION` on `win32`.
- **Arm 0 (`greedy_fsa`):** FSA-constrained greedy decode (`max_dim=1408`, `max_len=3072`) recording per-token top-4 log-probs and reliability margins.
- **Arm 1 (`orbgrand_sogrand`):** Reuses Arm 0's per-token log-probs; on any bar with `|duration_sum - bar_capacity| > 1e-4` (`syndrome != 0`), runs CPU ORBGRAND ordered-noise search (up to `64` logistic-weight patterns over the lowest-margin positions) until `syndrome == 0`, recording SOGRAND's analytical `p_not_found`, repair rate, and CPU ms/page.
- **Arm 2 (`trellis_bma_3view`):** On pages flagged by SOGRAND (`mean_p_not_found > 0.15` or `syndrome_violating_bars > 0`), decodes 3 cheap views (`base_1408`, `clahe_1408`, `scale_1280`), aligns bars, and fuses via Trellis-BMA IDS majority consensus with duration-syndrome tie-breaking.

## 4. Verification & Launch Commands
CPU mock smoke test (ORBGRAND repair + SOGRAND posterior + Trellis-BMA 3-view consensus):
```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe experiments\2026-11-27T14-55-i15-48-xf-orbgrand-syndrome-and-trellis-bma\run_colab.py --smoke-test
```
Colab A100 / Local 4080 SUPER run:
```bash
python3 -u experiments/2026-11-27T14-55-i15-48-xf-orbgrand-syndrome-and-trellis-bma/run_colab.py
```

## 5. Checkpointing & Remote HF Auto-Resume Plan
- On startup, checks HF Private Hub for `evals/<EXP_ID>.json` and `checkpoints/<EXP_ID>/{arm}_arm_result.json` and skips any already-completed arm.
- After each arm finishes, atomically writes local JSON and uploads both `checkpoints/<EXP_ID>/{arm}_arm_result.json` and partial `evals/<EXP_ID>.json` to HF Private Hub.
- Security: reads `HF_TOKEN` from `/tmp/.hf_token` (chmod 600), env `HF_TOKEN`, or `~/.cache/huggingface/token`, never prints it, always unlinks `/tmp/.hf_token` in `finally`.

## 6. Evaluation Gate & Decision Tree
- **Primary Metrics:** Token LCS F1 (`printed`, `hw`, `all12`), bar-duration syndrome violation rate (`syndrome_rate`), SOGRAND `mean_p_not_found` calibration, and latency overhead (`ms/page`).
- **PROMOTE:** `orbgrand_sogrand` reduces `syndrome_rate` by `>= 25%` relative and improves `all12` Token LCS F1 by `>= +0.005` at `< +5%` latency, AND/OR `trellis_bma_3view` adds `>= +0.010` F1 on flagged pages → port ORBGRAND+SOGRAND+Trellis-BMA into `eval_uni_suite31.py` and the Rust/ONNX inference pipeline.
- **ITERATE:** `syndrome_rate` drops strongly (`>= 25%`) but F1 is flat due to pitch/accidental swaps → restrict ORBGRAND flips to duration/dot/rest tokens with tighter logistic-weight cap or combine with KV-rewind.
- **DISCARD:** Neither arm improves Token LCS F1 (`< +0.002`) or repair rate is `< 10%` → drop cleanly.
