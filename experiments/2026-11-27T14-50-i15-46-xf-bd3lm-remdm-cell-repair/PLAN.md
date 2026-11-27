# PLAN — `2026-11-27T14-50-i15-46-xf-bd3lm-remdm-cell-repair`

## 1. Platform & Budget
- **Platform:** Google Colab `1× NVIDIA A100` / `H100` (or Local `RTX 4080 SUPER` 16 GB, native BF16).
- **Estimated Wall-Clock:** ~28 min total on A100 (2 arms × 600 steps + AR val CE + 2-step `ReMDM-conf` bidirectional repair eval + exact Hybrid INT4 packing & dequant eval).
- **Baseline Commit SHA:** `6055c49`.
- **Self-Containment:** All `BD3ReMDMRepairHead` and `ReMDM-conf` evaluation code lives strictly inside `run_colab.py`; `tmp-train/mvd/uni_omr_v2.py` is untouched.

## 2. Identifiers for Follow-Up Queries
- **Experiment ID:** `2026-11-27T14-50-i15-46-xf-bd3lm-remdm-cell-repair`
- **Short ID:** `i15-46` (`wave=0`, P0 cross-field)
- **Hugging Face (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-27T14-50-i15-46-xf-bd3lm-remdm-cell-repair.json`
  - Run Ledger: `runs/2026-11-27T14-50-i15-46-xf-bd3lm-remdm-cell-repair.json`
  - Partial Arm Sync: `checkpoints/2026-11-27T14-50-i15-46-xf-bd3lm-remdm-cell-repair/{ctrl,bd3_remdm_head}_arm_result.json`

## 3. Data & Controlled 2-Arm Protocol
- **Dataset:** `datasets/iter10_5_proxy_2k.tar` (`2,000` train / `64` val full pages, permissive licenses only, `0` `CC-BY-NC`, `2×` dense multi-staff & HW sampling weight, `FULLPAGE_MAX_DIM=1280`, `MAX_SEQ_LEN=1536`).
- **Shared Hyperparameters:** `600` steps/arm, `seed=42`, BF16 autocast, `batch_size=2`, `grad_accum=4` (eff. batch 8), **verified `muon+adamw`** (`lr=3e-4`, `weight_decay=0.04`, `no_decay_1d=True`; asserts `opt.mode == "muon+adamw"` on CUDA), WSD cosine schedule (`warmup=60`, `decay_frac=0.25`, `min_lr_ratio=0.05`), EMA (`decay=0.99` every step), `QATInt4ControllerV2` enabled at step `420` (final 30%), `grad_checkpoint=True`.
- **Windows RTX 4080 SUPER Safety:** Disables `torch.backends.cudnn.enabled` and forces CUTLASS `EFFICIENT_ATTENTION` on `win32`.
- **Arm 0 (`ctrl`):** Stock `UniOMRv2Config.edge85m(n_axial_layers=0, n_global_layers=6, grad_checkpoint=True)` causal AR decoder.
- **Arm 1 (`bd3_remdm_head`):** Same `edge85m + ax0_gl6` model + `BD3ReMDMRepairHead` (`d_rep=256`, `n_heads=4`, `d_ff=512`, `n_layers=2`, intra-cell `block_size=32` bidirectional self-attention with 1D RoPE + visual cross-attention to `mem_final`, sharing `tok_emb` and `lm_head`; `< 1.5 MB` INT4, total model `< 55.0 MB` INT4). Trained jointly with complementary masking (`15-40%` mask + `10%` single-token substitution corruption, loss weight `0.25`). QAT-INT4 covers `bd3_repair_head` Linear layers as well.

## 4. Verification & Launch Commands
CPU 2-step smoke test (local, zero GPU):
```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe experiments\2026-11-27T14-50-i15-46-xf-bd3lm-remdm-cell-repair\run_colab.py --smoke-test
```
Colab A100 / Local 4080 SUPER run:
```bash
python3 -u experiments/2026-11-27T14-50-i15-46-xf-bd3lm-remdm-cell-repair/run_colab.py
```

## 5. Checkpointing & Remote HF Auto-Resume Plan
- Atomic checkpoint save (`latest.pt.tmp` → `os.replace`) every `5.0` minutes per arm under `WORK_ROOT/ckpts/<arm>/` with full step/optimizer/EMA/RNG state.
- **1-Arm HF Partial Sync & Startup Auto-Resume:** On startup, checks HF Private Hub for `evals/<EXP_ID>.json` and `checkpoints/<EXP_ID>/{ctrl,bd3_remdm_head}_arm_result.json` and skips any already-completed arm. Uploads Arm 0 (`ctrl`) JSON + partial `evals/<EXP_ID>.json` (`status: "RUNNING"`) immediately before starting Arm 1 (`bd3_remdm_head`).
- Security: reads `HF_TOKEN` from `/tmp/.hf_token` (chmod 600), env `HF_TOKEN`, or `~/.cache/huggingface/token`, never prints it, always unlinks `/tmp/.hf_token` in `finally`.

## 6. Evaluation Gate & Decision Tree
- **Primary Metrics:** Standard AR `ema_val_ce` & `int4_dequant_val_ce`, 2-step `ReMDM-conf` bidirectional repair accuracy on corrupted spans (`remdm_corrupt_acc_step0/1/2`) and on lowest-20%-confidence AR-predicted spans (`remdm_ar_lowconf_acc_before` vs `after`), `step_ms`, and exact serialized `packed_int4_mb` (`< 55.0 MB` total, `< 1.5 MB` repair head overhead).
- **PROMOTE:** `bd3_remdm_head` preserves or improves AR `ema_val_ce` (`<= +0.005` vs `ctrl`), achieves `>= +15%` absolute accuracy gain on corrupted spans via 2-step `ReMDM-conf` AND positive net repair gain on low-confidence AR spans, stays `< 55.0 MB` INT4 (`< 1.5 MB` delta) → wire `BD3ReMDMRepairHead` into Iter 15 as an optional post-AR measure/cell repair pass.
- **ITERATE:** Strong corrupted-span infilling but weak on AR substitution errors (or slight AR CE drag `> +0.005`) → detach `mem_final` gradients from the repair head (`mem_final.detach()`) and increase single-token substitution noise ratio from `10%` to `25%`.
- **DISCARD:** AR `ema_val_ce` regresses by `> +0.020` or 2-step `ReMDM-conf` fails to improve low-confidence AR spans → drop cleanly.
