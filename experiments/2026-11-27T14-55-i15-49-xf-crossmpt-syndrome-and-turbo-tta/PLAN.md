# PLAN — `2026-11-27T14-55-i15-49-xf-crossmpt-syndrome-and-turbo-tta`

## 1. Platform & Budget
- **Platform:** Google Colab `1× NVIDIA A100` (or Local `RTX 4080 SUPER` 16 GB, native BF16).
- **Estimated Wall-Clock:** ~28 min total on A100 (2 arms × 600 steps + val CE + exact Hybrid INT4 packing & dequant eval + 3-step zero-label Turbo TTA on degraded val pages).
- **Baseline Commit SHA:** `6055c49`.
- **Self-Containment:** All `CrossMPTRefiner` and Turbo-TTA code lives strictly inside `run_colab.py`; `tmp-train/mvd/uni_omr_v2.py` is untouched.

## 2. Identifiers for Follow-Up Queries
- **Experiment ID:** `2026-11-27T14-55-i15-49-xf-crossmpt-syndrome-and-turbo-tta`
- **Hugging Face (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-27T14-55-i15-49-xf-crossmpt-syndrome-and-turbo-tta.json`
  - Run Ledger: `runs/2026-11-27T14-55-i15-49-xf-crossmpt-syndrome-and-turbo-tta.json`
  - Partial Arm Sync: `checkpoints/2026-11-27T14-55-i15-49-xf-crossmpt-syndrome-and-turbo-tta/{ctrl,crossmpt_turbo}_arm_result.json`

## 3. Data & Controlled 2-Arm Protocol
- **Dataset:** `datasets/iter10_5_proxy_2k.tar` (`2,000` train / `64` val full pages, permissive licenses only, `0` `CC-BY-NC`, `2×` dense multi-staff & HW sampling weight, `FULLPAGE_MAX_DIM=1280`, `MAX_SEQ_LEN=1536`).
- **Shared Hyperparameters:** `600` steps/arm, `seed=42`, BF16 autocast, `batch_size=2`, `grad_accum=4` (eff. batch 8), **verified `muon+adamw`** (`lr=3e-4`, `weight_decay=0.04`, `no_decay_1d=True`; asserts `opt.mode == "muon+adamw"` on CUDA), WSD cosine schedule (`warmup=60`, `decay_frac=0.25`, `min_lr_ratio=0.05`), EMA (`decay=0.99` every step), `QATInt4ControllerV2` enabled at step `420` (final 30%), `grad_checkpoint=True`.
- **Windows RTX 4080 SUPER Safety:** Automatically disables `torch.backends.cudnn.enabled` and forces CUTLASS `EFFICIENT_ATTENTION` on `win32`.
- **Arm 0 (`ctrl`):** Stock `UniOMRv2Config.edge85m(n_axial_layers=0, n_global_layers=6, grad_checkpoint=True)`.
- **Arm 1 (`crossmpt_turbo`):**
  - Attaches a tiny 2-layer `CrossMPTRefiner` (`d_ref=128`, `n_heads=4`, `< 1.5 MB` INT4): computes per-bar expected duration conservation syndromes `(sum_dur - cap, |sum_dur - cap|, sum_dur / cap, mean_entropy)` from base logits, projects them into syndrome keys/values, and runs 2 layers of bar-masked cross-attention from token logit-magnitude/hidden queries to their bar's syndrome embedding, producing a gated residual logit delta `logits + tanh(gate) * out_proj(ref)`.
  - At eval time, also runs 3-step zero-label syndrome-minimizing encoder-norm Turbo TTA (updating only encoder `RMSNorm`/`GroupNorm`/`LayerNorm` affine weights via Adam `lr=1e-3`, minimizing differentiable bar-conservation syndrome + entropy) on a degraded (synthetic blur+contrast+noise) validation subset, restoring weights after each page.

## 4. Verification & Launch Commands
CPU 2-step smoke test (local, zero GPU):
```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe experiments\2026-11-27T14-55-i15-49-xf-crossmpt-syndrome-and-turbo-tta\run_colab.py --smoke-test
```
Colab A100 / Local 4080 SUPER run:
```bash
python3 -u experiments/2026-11-27T14-55-i15-49-xf-crossmpt-syndrome-and-turbo-tta/run_colab.py
```

## 5. Checkpointing & Remote HF Auto-Resume Plan
- Atomic checkpoint save (`latest.pt.tmp` → `os.replace`) every `5.0` minutes per arm under `WORK_ROOT/ckpts/<arm>/`.
- **1-Arm HF Partial Sync & Startup Auto-Resume:** Checks HF Private Hub on startup for `evals/<EXP_ID>.json` and `checkpoints/<EXP_ID>/{ctrl,crossmpt_turbo}_arm_result.json`; uploads Arm 0 immediately upon completion before starting Arm 1.
- Security: reads `HF_TOKEN` from `/tmp/.hf_token` (chmod 600), env `HF_TOKEN`, or `~/.cache/huggingface/token`, never prints it, always unlinks `/tmp/.hf_token` in `finally`.

## 6. Evaluation Gate & Decision Tree
- **Primary Metrics:** `ema_val_ce`, `int4_dequant_val_ce`, `degraded_val_ce` vs `turbo_tta_degraded_val_ce`, `step_ms`, and exact serialized `packed_int4_mb` (`< 55.0 MB` hard budget; refiner delta `< 1.5 MB` INT4).
- **PROMOTE:** `crossmpt_turbo` beats `ctrl` by `>= -0.015` on `ema_val_ce` (and `int4_dequant_val_ce`), stays `< 55.0 MB` packed INT4 (`+refiner < 1.5 MB`), and Turbo TTA improves degraded val CE by `>= -0.010` → port `CrossMPTRefiner` into `uni_omr_v2.py` for Iter 15 and Turbo TTA into inference.
- **ITERATE:** `CrossMPTRefiner` helps (`-0.005` to `-0.015`) or Turbo TTA helps alone while refiner is neutral → decouple and promote the winning component independently.
- **DISCARD:** `< -0.005` gain on `ema_val_ce` and `< -0.005` gain from Turbo TTA → drop cleanly.
