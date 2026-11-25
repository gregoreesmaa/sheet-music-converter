# Experiment Plan — `2026-11-26T00-54-ablation-unlabeled-rism-rlvr`

## 1. Platform Chosen & Rationale
- **Platform:** **Modal `L4` (`24 GB VRAM`, native `BF16`, `cc=8.9`)** via `modal.App("exp5-ablation-unlabeled-rism-rlvr")`.
- **Rationale:** GRPO with `G=4` rollouts + frozen `ref_model` needs comfortable BF16 VRAM headroom (`24 GB`) and runs fast on `L4` (`~$0.80/hr`). Keeps the local RTX 4080 SUPER completely untouched (`CUDA_VISIBLE_DEVICES=""`).
- **Estimated Wall-Clock & Cost:** `~18–25 min` total (`~5 min` before+after 45-piece `suite31_hw` eval + `~12–15 min` for `120` GRPO updates) → **`~$0.25–$0.35`** out of the `$30/mo` free tier. Function timeout set to `7200s` (2 hours).

## 2. Identifiers for Follow-Up Queries (Zero-Prior-Context Handoff)
- **Modal Profile:** `gregoreesmaa1`
- **Modal App Name:** `exp5-ablation-unlabeled-rism-rlvr`
- **Modal App ID:** `ap-ozqdCx4lsObGII8YLKdwFV`
- **Modal Function Call ID:** `fc-01M8AG6ZSQ1ND65K1EK8M2WSCY`
- **Modal Dashboard URL:** `https://modal.com/apps/gregoreesmaa1/main/ap-ozqdCx4lsObGII8YLKdwFV`
- **Modal Volume:** `omr-cache` mounted at `/vol` (checkpoints at `/vol/ckpts/exp5_rism_rlvr/`)
- **Modal Secret:** `hf-secret` (`HF_TOKEN`)
- **Hugging Face Repo:** `gregoreesmaa/sheet-music-converter-omr` (private dataset)
  - Ledger: `runs/2026-11-26T00-54-ablation-unlabeled-rism-rlvr.json`
  - Final checkpoint: `checkpoints/2026-11-26T00-54-ablation-unlabeled-rism-rlvr/uni_omr_rlvr_final.pt`
  - Eval JSON: `evals/2026-11-26T00-54-ablation-unlabeled-rism-rlvr.json`

### Copy-Paste Monitoring / Control Commands (PowerShell, Repo Root)
```powershell
# Check status of all Modal apps
.\.venv\Scripts\modal.exe app list

# Tail live container logs
.\.venv\Scripts\modal.exe app logs ap-ozqdCx4lsObGII8YLKdwFV

# Stop runaway job if needed
.\.venv\Scripts\modal.exe app stop ap-ozqdCx4lsObGII8YLKdwFV

# Inspect Hugging Face run ledger & eval once complete
.\.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; p = hf_hub_download('gregoreesmaa/sheet-music-converter-omr', 'runs/2026-11-26T00-54-ablation-unlabeled-rism-rlvr.json', repo_type='dataset'); print(open(p).read())"
```

## 3. Required Training & Eval Data (100% Permissive, Zero Eval Overlap)
- **Unlabeled Real HW Corpus:** `datasets/rism_hw_5707.tar` (`3.03 GB`, `5,707` unlabeled historical RISM handwritten manuscript pages, permissive public-domain archival scans, zero `suite31_hw` overlap). Cached to `/vol/datasets/rism_hw/`.
- **Evaluation Benchmark:** `datasets/suite31_hw_eval.tar` (`130.8 MB`, 45 pieces: 31 printed + 14 real historical handwritten manuscripts, `clean` + `photo` = 90 pages via `suite31_hw_manifest.json`). Cached to `/vol/datasets/suite31_hw_eval/`.
- **Baseline Checkpoint (`--init-from`):** `checkpoints/iter10_soup_baseline/uni_omr_soup.pt` (`179.2 MB`, Iter-10 v1 soup; local Rust Strict Note F1 baseline: `hw_clean 0.2550`, `printed_clean 0.8895`).

## 4. Exact Launch Command & Hyperparameters (Commit `9d73170`)
```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\modal.exe run --detach experiments/2026-11-26T00-54-ablation-unlabeled-rism-rlvr/run_modal.py
```
- **GRPO Hyperparameters (`GRPOConfig` in `tmp-train/mvd/rlvr_grpo.py`):**
  - `updates=120`, `group_size=4`, `max_len=384`, `max_dim=1024`, `unlabeled_frac=1.0`
  - `temperature=0.8`, `top_p=0.95`, `clip_eps=0.2`, `kl_beta=0.04`
  - `use_grammar=True`, `strict_grammar=False`, `unlabeled_vocab=id_to_tok` (includes `<sys>`)
  - Optimizer: `AdamW(lr=1e-5, weight_decay=0.01)` with 10-step linear warmup + cosine decay to `2e-6`, `clip_grad_norm_(1.0)`, `BF16` native autocast.
- **Eval Hyperparameters:** All 45 `suite31_hw` pieces (`clean` + `photo` = 90 pages), `max_dim=1024`, `max_len=1536`, `MTP-4 + FSA grammar`, `BF16`, run **before** and **after** RLVR under the exact same evaluator, plus saved `hyp_bekern` strings for optional local Rust `eval-harness.exe` Strict Note F1 verification.

## 5. 5-Minute Checkpointing & Resume Plan
- **Atomic 5-Min Checkpoints:** Every `>= 5.0` minutes (`save_every_minutes=5.0`), `run_modal.py` writes `/vol/ckpts/exp5_rism_rlvr/uni_omr_rlvr_latest.pt.tmp` → `os.replace(..., uni_omr_rlvr_latest.pt)` (model + optimizer + step + history) and calls `vol.commit()`.
- **HF Sync:** Run ledger `runs/2026-11-26T00-54-ablation-unlabeled-rism-rlvr.json` updates on start, every 15 min, and on completion; final checkpoint (`checkpoints/.../uni_omr_rlvr_final.pt`) and full eval (`evals/...json`) upload on completion.
- **Auto-Resume Command:** Simply re-run the exact same launch command — `run_modal.py` automatically detects `/vol/ckpts/exp5_rism_rlvr/uni_omr_rlvr_latest.pt` and `/vol/ckpts/exp5_rism_rlvr/baseline_eval.json` and resumes from the last completed step.

## 6. Evaluation Gate & Decision Tree ("What We Do With the Result")
- **Primary Metrics (Apples-to-Apples Before vs After on 45-piece `suite31_hw`):**
  - `hw_clean` & `hw_photo`: Python Exact Pitch+Duration Note F1 (`note_f1_py_mean`), Structural Validity Reward (`struct_reward_mean`), Bar-Duration Conservation (`r_rhythm_mean`), Anti-Loop (`r_noloop_mean`).
  - `printed_clean` & `printed_photo`: Regression guard on Note F1 and Structural Validity.
  - Optional local verification: download `checkpoints/2026-11-26T00-54-ablation-unlabeled-rism-rlvr/uni_omr_rlvr_final.pt` and run `eval_uni_suite31.py` for official Rust `eval-harness.exe` Strict Note F1.
- **Promote:** If `hw_clean` or `hw_photo` `note_f1_py_mean` improves by **`>= +0.015`** (or `r_rhythm_mean` improves `>= +0.05` with non-negative Note F1) AND `printed_clean` drops by **`<= 0.005`** → integrate label-free RISM RLVR post-training as a standard post-SFT stage after Iter-10.5 / Iter-11.
- **Iterate:** If `struct_reward` / `r_rhythm` improves strongly on HW but `note_f1_py` is flat or printed drifts `> 0.005` → spawn a follow-up mixing `unlabeled_frac=0.30` with labeled `iter10_5_proxy_2k` (`compute_omr_rlvr_reward`) and/or higher `kl_beta=0.08`.
- **Discard:** If neither HW Note F1 nor HW structural validity improves (or reward hacking collapses output length / `H/R` ratio) → document in `OUTCOME.md` and rely on supervised `CUT-RISM` + multi-view pseudo-labels instead.
