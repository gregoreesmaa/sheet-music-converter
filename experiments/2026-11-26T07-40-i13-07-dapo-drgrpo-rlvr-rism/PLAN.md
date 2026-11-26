# Experiment Plan — `2026-11-26T07-40-i13-07-dapo-drgrpo-rlvr-rism`

> **Zero-Prior-Context Handoff:** Everything required to run, monitor, stop, resume, or evaluate `i13-07` (DAPO + Dr. GRPO unbiased RLVR ablation on unlabeled RISM) is self-contained in this folder (`README.md`, `PLAN.md`, `run_colab.py`).

## 1. Platform & Budget
- **Provider:** Google Colab `1× NVIDIA L4` (`24 GB`, `cc=8.9`, native BF16) via `.\.venv\Scripts\colab.exe`.
- **Session Name:** `i13-07-dapo-drgrpo`
- **Estimated Wall-Clock & Cost:** `~38 min` (`~3.1 CU`):
  - Baseline eval (`14` HW + `12` stratified Printed at `1280px / 3840`): `~3 min`
  - `4` arms × `120` RLVR updates (`G=8`, `max_len=512`, `max_dim=1024`, `kl_beta=0.06`): `~24 min`
  - `4` post-arm evals (`26` pieces at `1280px / 3840`): `~11 min`

## 2. Identifiers
- **Experiment ID:** `2026-11-26T07-40-i13-07-dapo-drgrpo-rlvr-rism`
- **HF Repo:** `gregoreesmaa/sheet-music-converter-omr` (`repo_type="dataset"`)
  - Final Eval JSON: `evals/2026-11-26T07-40-i13-07-dapo-drgrpo-rlvr-rism.json`
  - Run Ledger JSON: `runs/2026-11-26T07-40-i13-07-dapo-drgrpo-rlvr-rism.json`
- **Base Checkpoint:** `checkpoints/iter10_5_best/uni_omr_ema_best.pt` (fallback: `checkpoints/iter10_soup_baseline/uni_omr_soup.pt`)
- **Datasets:** `datasets/rism_hw_5707.tar` (unlabeled RISM HW pages) + `datasets/suite31_hw_eval.tar` (45-piece `suite31_hw`).

## 3. Exact Launch Commands (PowerShell)
```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\colab.exe new -s i13-07-dapo-drgrpo --gpu L4
.\.venv\Scripts\colab.exe upload -s i13-07-dapo-drgrpo <local_bundle.tar.gz> /content/bundle.tar.gz
.\.venv\Scripts\colab.exe upload -s i13-07-dapo-drgrpo <local_tmp_token> /tmp/.hf_token
.\.venv\Scripts\colab.exe exec -s i13-07-dapo-drgrpo -f experiments/2026-11-26T07-40-i13-07-dapo-drgrpo-rlvr-rism/run_colab.py
.\.venv\Scripts\colab.exe stop -s i13-07-dapo-drgrpo
```

## 4. Hyperparameters (Held Fixed Across All 4 Arms)
- `updates = 120`, `group_size = 8`, `kl_beta = 0.06`, `rollout_max_len = 512`, `rollout_max_dim = 1024`
- `temperature = 0.85`, `top_p = 0.95`, `lr = 1.0e-5` (warmup `12` + cosine to `2.0e-6`), `seed = 1307`
- Canonical eval: `max_dim = 1280`, `eval_max_len = 3840`, `use_grammar = True`, per-piece `note_f1_py` + `tok_f1_py`.
- Tracked diagnostics per arm: `mean_policy_entropy`, `mean_rollout_len`, `zero_var_skipped`, `clip_frac`, `mean_kl`, `n_hw_improved / 14`.

## 5. Checkpointing & Resume
- Every `5.0` minutes during an arm, saves atomic `latest.pt` + `state.json` under `/content/cache/ckpts/<EXP_ID>/<arm>/`.
- Completed arms cache `arm_result.json` so interrupted sessions resume cleanly.

## 6. Promotion Gate & Decision Tree
- **PROMOTE:** Best non-vanilla arm (`dr_grpo`, `dapo_clip_dyn`, or `dapo_drgrpo_hybrid`) beats `vanilla_grpo` by `>= +0.015` on `hw_clean` OR `hw_photo` Note F1 AND improves `>= 8/14` HW pieces with `printed_clean` drop `<= 0.005` AND maintains higher policy entropy without length inflation → promote winning RLVR objective into `rlvr_grpo.py` for Iter 13 post-training.
- **ITERATE:** Gains appear on `hw_clean` but regress on `hw_photo` or show KL drift → combine winning objective with `dual_verifier.py` (`R_sym + R_vis`) in Iter 13.
- **DISCARD:** All 3 variants tie or trail `vanilla_grpo` within `±0.005` → retain vanilla GRPO in `rlvr_grpo.py`.
