# PLAN — `2026-11-28T00-26-i15-48b-xf-orbgrand-syndrome-trellis-fixed`

## 1. Platform & Budget
- **Platform:** Google Colab `1× NVIDIA A100` (or Local `RTX 4080 SUPER` 16 GB, native BF16).
- **Estimated Wall-Clock:** ~10 min total (zero training; 12 hard `suite31_hw` pages × 3 inference arms).
- **Baseline Commit SHA:** `dddf9a2`.
- **Predecessor:** `2026-11-27T14-55-i15-48-xf-orbgrand-syndrome-and-trellis-bma` (`ITERATE` due to leading `voice-1` early-`break` bug in `compute_bar_duration_syndrome`).
- **Self-Containment:** All ORBGRAND, SOGRAND, Trellis-BMA, and MusicXML scoring code lives strictly inside `run_colab.py`.

## 2. Identifiers for Follow-Up Queries
- **Experiment ID:** `2026-11-28T00-26-i15-48b-xf-orbgrand-syndrome-trellis-fixed`
- **Hugging Face (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-28T00-26-i15-48b-xf-orbgrand-syndrome-trellis-fixed.json`
  - Run Ledger: `runs/2026-11-28T00-26-i15-48b-xf-orbgrand-syndrome-trellis-fixed.json`
  - Partial Arm Sync: `checkpoints/2026-11-28T00-26-i15-48b-xf-orbgrand-syndrome-trellis-fixed/{greedy_fsa,orbgrand_sogrand,trellis_bma_3view}_arm_result.json`

## 3. Data, Checkpoint & 3-Arm Inference Protocol
- **Checkpoint Resolution (auto-fallback):** Checks `UNIOMR_CKPT` env → HF `checkpoints/iter14-v2a-final/uni_omr_v2_ema_latest.pt` → `checkpoints/iter14-final/uni_omr_v2_ema_latest.pt` → `checkpoints/iter14-v2a-mid/uni_omr_v2_ema_latest.pt` → `checkpoints/iter10_soup_baseline/uni_omr_soup.pt`.
- **Eval Subset:** Deterministic 12-page hard printed + HW subset of `datasets/suite31_hw_eval.tar` (6 hard printed `photo` pages + 6 hard real handwritten `photo`/`clean` pages; permissive/benchmark provenance, zero `CC-BY-NC`).
- **Windows RTX 4080 SUPER Safety:** Automatically disables `torch.backends.cudnn.enabled` and forces CUTLASS `EFFICIENT_ATTENTION` on `win32`.
- **Fixed Multi-Staff / Multi-Voice Syndrome (`compute_bar_duration_syndrome`):**
  - Segments each bar by `(staff_idx, voice_id)` on `<staff>` / `staff-*` and `voice-*` tokens instead of `break`ing.
  - Computes per-segment beat sum vs `bar_capacity` (`bar-rest` = full capacity for that segment, `chord` = 0 advance, `dot` = `1.5×` chained, `tuplet-start-A-N`...`tuplet-stop` scaling, `grace` = 0 advance).
  - Returns `is_valid = False` and non-zero aggregate syndrome whenever **any** rhythmic `(staff, voice)` segment violates `bar_capacity`.
- **Arm 0 (`greedy_fsa`):** FSA-constrained greedy decode (`max_dim=1408`, `max_len=3072`) recording per-token top-4 probs and `hit_eos`.
- **Arm 1 (`orbgrand_sogrand`):** Reuses Arm 0's per-token probs; on any bar with non-zero `(staff, voice)` syndrome, runs CPU ORBGRAND ordered-noise search (up to `64` logistic-weight patterns over up to `12` lowest-margin rhythm positions) until all segments satisfy `bar_capacity`, recording SOGRAND's analytical `p_not_found`, repair count, and CPU ms/page.
- **Arm 2 (`trellis_bma_3view`):** On pages flagged by SOGRAND (`mean_p_not_found > 0.15` or `n_viol_after > 0`), decodes 3 cheap views (`base_1408`, `clahe_1408`, `scale_1280`), aligns bars, and fuses via Trellis-BMA IDS majority consensus with multi-voice duration-syndrome tie-breaking.

## 4. Verification & Launch Commands
CPU smoke test (multi-voice/multi-staff syndrome, ORBGRAND repair, SOGRAND flagging, Trellis-BMA 3-view consensus, MusicXML scorer):
```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe experiments\2026-11-28T00-26-i15-48b-xf-orbgrand-syndrome-trellis-fixed\run_colab.py --smoke-test
```
Colab A100 / Local 4080 SUPER run:
```bash
python3 -u experiments/2026-11-28T00-26-i15-48b-xf-orbgrand-syndrome-trellis-fixed/run_colab.py
```

## 5. Checkpointing & Remote HF Auto-Resume Plan
- On startup, checks HF Private Hub for `evals/<EXP_ID>.json` and `checkpoints/<EXP_ID>/{arm}_arm_result.json` and skips any already-completed arm.
- After each arm finishes, atomically writes local JSON and uploads both `checkpoints/<EXP_ID>/{arm}_arm_result.json` and partial `evals/<EXP_ID>.json` to HF Private Hub.
- Security: reads `HF_TOKEN` from `/tmp/.hf_token` (chmod 600), env `HF_TOKEN`, or `~/.cache/huggingface/token`, never prints it, always unlinks `/tmp/.hf_token` in `finally`.

## 6. Evaluation Gate & Decision Tree
- **Primary Metrics:** MusicXML `note_f1` (`printed`, `hw`, `all12`), `tok_lcs_f1`, `bsvr`, `wfpp`, multi-voice `syndrome_rate`, SOGRAND `mean_p_not_found`, latency overhead (`ms/page`), and explicit `gate_pass`.
- **PROMOTE:** `orbgrand_sogrand` reduces `syndrome_rate` by `>= 25%` relative and improves `all12` `token_f1` or `note_f1` by `>= +0.005` at `< +5%` latency, AND/OR `trellis_bma_3view` adds `>= +0.010` `token_f1` or `note_f1` on flagged pages → port ORBGRAND+SOGRAND+Trellis-BMA into `eval_uni_suite31.py` and the Rust/ONNX inference pipeline.
- **ITERATE:** `syndrome_rate` drops strongly (`>= 25%`) but F1 is flat due to pitch/accidental swaps → tighten rhythm-only flips or combine with KV-rewind.
- **DISCARD:** Neither arm improves F1 (`< +0.002`) or repair rate is `< 10%` → drop cleanly.
