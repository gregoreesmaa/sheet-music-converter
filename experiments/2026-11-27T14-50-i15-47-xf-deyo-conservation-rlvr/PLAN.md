# PLAN — `2026-11-27T14-50-i15-47-xf-deyo-conservation-rlvr`

## 1. Platform & Budget
- **Platform:** Google Colab `1× NVIDIA A100` (or Local `RTX 4080 SUPER` 16 GB, native BF16).
- **Estimated Wall-Clock:** ~28–35 min total on A100 (Arm 0: 600 SFT steps; Arm 1: 450 SFT + 150 DeYO+GRPO steps + val CE + 16-page rollout reward eval + exact INT4 pack).
- **Baseline Commit SHA:** `6055c49`.
- **Self-Containment:** All DeYO PLPD gating and short-horizon GRPO loop logic lives strictly inside `run_colab.py` (reusing `rlvr_grpo.compute_omr_rlvr_reward`); `tmp-train/mvd/` is untouched.

## 2. Identifiers for Follow-Up Queries
- **Experiment ID:** `2026-11-27T14-50-i15-47-xf-deyo-conservation-rlvr`
- **Hugging Face (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-27T14-50-i15-47-xf-deyo-conservation-rlvr.json`
  - Run Ledger: `runs/2026-11-27T14-50-i15-47-xf-deyo-conservation-rlvr.json`
  - Partial Arm Sync: `checkpoints/2026-11-27T14-50-i15-47-xf-deyo-conservation-rlvr/{sft_ctrl,deyo_rlvr}_arm_result.json`

## 3. Data & Controlled 2-Arm Protocol
- **Dataset:** `datasets/iter10_5_proxy_2k.tar` (`2,000` train / `64` val full pages, permissive licenses only, zero `CC-BY-NC`, `2×` dense multi-staff & HW sampling weight, `FULLPAGE_MAX_DIM=1280`, `MAX_SEQ_LEN=1536`).
- **Shared Hyperparameters:** `600` total steps/arm, `seed=42`, BF16 autocast, `batch_size=2`, `grad_accum=4` (eff. batch 8), **verified `muon+adamw`** (`lr=3e-4`, `weight_decay=0.04`, `no_decay_1d=True`; asserts `opt.mode == "muon+adamw"` on CUDA), WSD cosine schedule (`warmup=60`, `decay_frac=0.25`, `min_lr_ratio=0.05`), EMA (`decay=0.99` every step), `QATInt4ControllerV2` enabled at step `420`, `grad_checkpoint=True`.
- **Windows RTX 4080 SUPER Safety:** Automatically disables `torch.backends.cudnn.enabled` and forces CUTLASS `EFFICIENT_ATTENTION` on `win32`.
- **Arm 0 (`sft_ctrl`):** 600 steps of standard SFT (`UniOMRv2Config.edge85m(n_axial_layers=0, n_global_layers=6)`).
- **Arm 1 (`deyo_rlvr`):**
  - Steps `1..450`: Identical SFT (shares exact same seed/schedule/QAT).
  - At step `450`: Snapshot frozen reference weights (`ref_state_dict`) for `k3` KL anchoring.
  - Steps `451..600` (`150` steps): Combined objective `0.50 * L_sft + 0.25 * L_deyo + 0.25 * L_grpo`:
    - `L_deyo`: Generate lightly degraded + patch-shuffled view of the page; compute DeYO PLPD (Pseudo-Label Probability Difference) to gate token-level symmetric KL consistency on visually grounded tokens.
    - `L_grpo`: Sample `G=4` KV-cached rollouts (`max_len=192`, `T=0.8`) per page, score with `rlvr_grpo.compute_omr_rlvr_reward` (exact syntax + bar-duration conservation + anti-loop + token/note edit similarity), compute group-normalized advantages `A_i = (R_i - mean(R)) / (std(R) + 1e-6)`, and apply clipped GRPO policy gradient + `k3` KL penalty (`kl_beta=0.02`).

## 4. Verification & Launch Commands
CPU 4-step smoke test (2 SFT + 2 DeYO+GRPO steps, zero GPU):
```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe experiments\2026-11-27T14-50-i15-47-xf-deyo-conservation-rlvr\run_colab.py --smoke-test
```
Colab A100 / Local 4080 SUPER run:
```bash
python3 -u experiments/2026-11-27T14-50-i15-47-xf-deyo-conservation-rlvr/run_colab.py
```

## 5. Checkpointing & Remote HF Auto-Resume Plan
- Atomic checkpoint save (`latest.pt.tmp` → `os.replace`) every `5.0` minutes per arm under `WORK_ROOT/ckpts/<arm>/`.
- **1-Arm HF Partial Sync & Startup Auto-Resume:** On startup, checks HF Private Hub for `evals/<EXP_ID>.json` and `checkpoints/<EXP_ID>/{sft_ctrl,deyo_rlvr}_arm_result.json` and skips completed arms. Uploads Arm 0 (`sft_ctrl`) results and partial `evals/<EXP_ID>.json` (`status: "RUNNING"`) to HF immediately before starting Arm 1.
- Security: reads `HF_TOKEN` from `/tmp/.hf_token` (chmod 600), `HF_TOKEN` env, or `~/.cache/huggingface/token`, never prints it, always unlinks `/tmp/.hf_token` in `finally`.

## 6. Evaluation Gate & Decision Tree
- **Primary Metrics:** `val_rlvr_reward`, `val_r_meter` (bar-duration conservation), `val_r_note` (edit similarity), `ema_val_ce`, `int4_dequant_val_ce`, and exact serialized `packed_int4_mb` (`< 55.0 MB`).
- **PROMOTE:** `deyo_rlvr` improves `val_rlvr_reward` by `>= +0.025` (with `val_r_meter` and `val_r_note` both up) while keeping `ema_val_ce` within `<= +0.03` of `sft_ctrl` and `packed_int4_mb < 55.0 MB` → adopt DeYO+GRPO as the Iter 15 Stage-3 post-training recipe.
- **ITERATE:** Improves `val_r_meter` cleanly (`>= +0.05`) but `ema_val_ce` drifts `> +0.03` or `val_r_note` is flat → increase `kl_beta` (`0.05`) / lower RLVR LR (`0.2×` SFT LR) or run GRPO on longer rollouts (`max_len=512`) from a converged checkpoint.
- **DISCARD:** `< +0.010` gain on `val_rlvr_reward` or reward-hacking collapse → keep standard SFT + inference-time verification.
