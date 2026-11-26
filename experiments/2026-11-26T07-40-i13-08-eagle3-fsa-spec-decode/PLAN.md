# Experiment Plan — `2026-11-26T07-40-i13-08-eagle3-fsa-spec-decode`

> **Zero-Prior-Context Handoff:** Everything required to run, monitor, stop, resume, or evaluate `i13-08` (EAGLE-3 multi-layer speculative drafter + FSA-constrained decoding vs MTP-4) is self-contained in this folder (`README.md`, `PLAN.md`, `run_colab.py`).

## 1. Platform & Budget
- **Provider:** Google Colab `1× NVIDIA L4` (`24 GB`, `cc=8.9`, native BF16) via `.\.venv\Scripts\colab.exe`.
- **Session Name:** `i13-08-eagle3-fsa`
- **Estimated Wall-Clock & Cost:** `~26 min` (`~2.1 CU`):
  - Shared `UniOMRv2Net(small_45m)` base training (`600` steps on `iter10_5_proxy_2k.tar`): `~7 min`
  - `Eagle3DraftHead` frozen-target TTT distillation (`300` steps, `K_draft=4`, `decay=0.85`): `~6 min`
  - 5-arm lossless speculative decode benchmark (`16` real `suite31_hw` pages: `8` printed + `8` HW, `max_dim=1280`, `max_len=512`, `B=1`): `~12 min`

## 2. Identifiers
- **Experiment ID:** `2026-11-26T07-40-i13-08-eagle3-fsa-spec-decode`
- **HF Repo:** `gregoreesmaa/sheet-music-converter-omr` (`repo_type="dataset"`)
  - Final Eval JSON: `evals/2026-11-26T07-40-i13-08-eagle3-fsa-spec-decode.json`
  - Run Ledger JSON: `runs/2026-11-26T07-40-i13-08-eagle3-fsa-spec-decode.json`
- **Datasets:** `datasets/iter10_5_proxy_2k.tar` (base + TTT training) + `datasets/suite31_hw_eval.tar` (16 real benchmark pages).

## 3. Exact Launch Commands (PowerShell)
```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\colab.exe new -s i13-08-eagle3-fsa --gpu L4
.\.venv\Scripts\colab.exe upload -s i13-08-eagle3-fsa <local_bundle.tar.gz> /content/bundle.tar.gz
.\.venv\Scripts\colab.exe upload -s i13-08-eagle3-fsa <local_tmp_token> /tmp/.hf_token
.\.venv\Scripts\colab.exe exec -s i13-08-eagle3-fsa -f experiments/2026-11-26T07-40-i13-08-eagle3-fsa-spec-decode/run_colab.py
.\.venv\Scripts\colab.exe stop -s i13-08-eagle3-fsa
```

## 4. Hyperparameters
- Target model: `UniOMRv2Config.small_45m(grad_checkpoint=True, mtp_k=4)`, `600` steps (`B=2, ga=4, lr=3e-4, max_dim=1280, max_seq_len=1536`).
- EAGLE-3 drafter: `Eagle3DraftHead.from_target_model(model)`, `300` TTT steps (`B=2, ga=4, lr=5e-4, K_draft=4, step_decay=0.85, temp=1.0`, target frozen).
- Eval: `16` real `suite31_hw` pages (`8` printed clean/photo + `8` HW clean/photo), `B=1`, `max_len=512`, `CUDA synchronize()` wall-clock timing after 1 warmup page.

## 5. Checkpointing & Resume
- Saves `/content/vol/ckpts/<EXP_ID>/base_fp_600.pt` and `eagle3_ttt_300.pt` atomically every `5.0` minutes; completed decode arms cache `arm_<name>.json`.

## 6. Promotion Gate & Decision Tree
- **PROMOTE:** `eagle3_k4_fsa` or `eagle3_k6_fsa` achieves **100% bit-exact greedy parity** (`exact_match_rate == 1.0`), beats `mtp4_fsa` on `mean_accepted_tokens_per_step` by `>= +0.25`, and achieves `>= 1.35×` lossless speedup over `greedy_fsa_baseline` at `< 1.5 MB` INT4 weight overhead → adopt EAGLE-3 + FSA drafter as Iter 13 default browser/server decoder (and optionally drop MTP-4 auxiliary heads to save `~2 MB`!).
- **ITERATE:** `eagle3_k4_fsa` beats `eagle3_k4_no_fsa` on acceptance rate but trails on wall-clock due to Python FSA overhead → move FSA draft mask into vectorized lookup / Rust WASM before promoting.
- **DISCARD:** `mtp4_fsa` matches or beats EAGLE-3 on both acceptance rate and `tok/s` → keep built-in MTP-4 in Iter 13.
