# Experiment Plan — `2026-11-26T00-55-trial-render-diff-rlvr`

## 1. Platform Chosen & Rationale
- **Platform:** **Modal `L4` (`24 GB VRAM`, `cc=8.9`, native BF16)** — `~$0.80/hr` within the `$30/mo` free Starter credit.
- **Rationale:** Render-and-diff GRPO runs `G=4` KV-cached rollouts + CPU Verovio/resvg rendering + `assembly-serialize` MusicXML conversion per step, plus full before/after 45-piece `suite31_hw` OOD evaluation. L4 gives 24 GB VRAM with native BF16 and keeps the local RTX 4080 SUPER completely untouched for Iter 10.5.
- **Estimated Wall-Clock & Cost:** `~25–45 min` (`~$0.35–$0.60` of `$30/mo` free credit). Hard timeout: `7200s` (2 hours).

## 2. Identifiers for Follow-Up Queries (Zero-Prior-Context Handoff)
- **Modal Profile:** `gregoreesmaa1`
- **Modal App Name:** `exp7-trial-render-diff-rlvr`
- **Modal App ID:** `ap-oLzdkXJmNcYxbzj8bkCLKT`
- **Modal Dashboard URL:** `https://modal.com/apps/gregoreesmaa1/main/ap-oLzdkXJmNcYxbzj8bkCLKT`
- **Modal Volume:** `omr-cache` mounted at `/vol` (checkpoints at `/vol/ckpts/exp7_render_diff_rlvr/`)
- **Modal Secret:** `hf-secret` (`HF_TOKEN`)
- **Hugging Face Repo:** `gregoreesmaa/sheet-music-converter-omr` (private dataset repo)
  - Final Checkpoint: `checkpoints/2026-11-26T00-55-trial-render-diff-rlvr/uni_omr_final.pt`
  - Rolling Checkpoint: `checkpoints/2026-11-26T00-55-trial-render-diff-rlvr/uni_omr_latest.pt`
  - Before/After Eval JSON: `evals/2026-11-26T00-55-trial-render-diff-rlvr.json`
  - Run Ledger JSON: `runs/2026-11-26T00-55-trial-render-diff-rlvr.json`

### Copy-Paste Monitoring & Control Commands (PowerShell, Repo Root)
```powershell
# List apps and check live state
.\.venv\Scripts\modal.exe app list

# Tail live logs
.\.venv\Scripts\modal.exe app logs ap-oLzdkXJmNcYxbzj8bkCLKT

# Stop/cancel runaway run
.\.venv\Scripts\modal.exe app stop ap-oLzdkXJmNcYxbzj8bkCLKT

# Download final eval & run ledger from Hugging Face once completed
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('gregoreesmaa/sheet-music-converter-omr', 'evals/2026-11-26T00-55-trial-render-diff-rlvr.json', repo_type='dataset'))"
```

## 3. Required Training & Eval Data (100% Permissive, 0 Eval Overlap)
- **Unlabeled RLVR Pages:** `datasets/iter10_5_proxy_2k.tar` on HF (`2,000` deduplicated permissive pages; labels are NOT passed to the reward function — only page images are used).
- **OOD Evaluation Suite:** `datasets/suite31_hw_eval.tar` on HF (`45` pieces: `31` printed + `14` historical handwritten manuscripts, `clean` + `photo` = `90` pages via `suite31_hw_manifest.json`).
- **Warm-Start Baseline Checkpoint:** `checkpoints/iter10_soup_baseline/uni_omr_soup.pt` on HF (`printed_clean 0.8895`, `printed_photo 0.8738`, `hw_clean 0.2550`, `hw_photo 0.1516`, `all45_clean 0.6921`).
- **In-Container Rust Binaries:** `converter/assembly-serialize` (via sibling `mscz-rs`) and `converter/eval-harness` compiled `--release` inside the cached Modal image layer.

## 4. Exact Launch Command & Hyperparameters (Commit `9d73170`)
```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\modal.exe run --detach experiments/2026-11-26T00-55-trial-render-diff-rlvr/run_modal.py
```
- **GRPO Hyperparameters:**
  - `steps = 60`, `batch_size = 1`, `group_size = 4` (`240` render-and-diff rollouts total)
  - `max_len = 512`, `train_max_dim = 1024`, `temperature = 0.8`, `top_p = 0.95`
  - `lr = 2e-5` (AdamW, `weight_decay = 0.01`, `clip_grad_norm = 1.0`), `kl_beta = 0.04`
  - `use_grammar = True`, `strict_grammar = False`, `bf16 = True` (cuDNN SDPA disabled to avoid variable-shape stalls)
- **Before & After `suite31_hw` Eval Settings:**
  - `--max-dim 1536 --max-len 2048 --use-grammar --use-mtp-spec --bf16`

## 5. 5-Minute Checkpointing & Resume Plan
- Every `>= 5.0` minutes (`save_every_minutes=5.0`), `run_modal.py` atomically writes `/vol/ckpts/exp7_render_diff_rlvr/uni_omr_latest.pt` (`.tmp` → `os.replace`) + run ledger JSON and commits `vol.commit()`.
- Every `>= 15` minutes and on completion, checkpoints + ledger sync to HF `gregoreesmaa/sheet-music-converter-omr`.
- **Resume Command:** Re-running `.\.venv\Scripts\modal.exe run --detach experiments/2026-11-26T00-55-trial-render-diff-rlvr/run_modal.py` automatically detects cached `suite31_hw_before.json` and `/vol/ckpts/exp7_render_diff_rlvr/uni_omr_latest.pt` (`resume=True`) and resumes from the last completed GRPO step.

## 6. Evaluation Gate & Decision Tree ("What We Do With the Result")
- **Primary Metrics:** Before vs. After `suite31_hw` Strict Note F1 (`printed_clean`, `printed_photo`, `hw_clean`, `hw_photo`, `all45_clean`, `all45_photo`), `decode_ok_rate`, and GRPO `mean_r_visual` / `render_ok_rate` trajectory.
- **PROMOTE:** If `all45_clean` or `all45_photo` Note F1 improves by `>= +0.005` with no slice regressing by `> 0.005` AND `render_ok_rate >= 0.70` → promote Render-and-Diff RLVR into our post-Iter-10.5 / Iter-11 unlabeled RISM (`5,707` pages) post-training stage!
- **ITERATE:** If `render_ok_rate >= 0.50` and `mean_r_visual` rises cleanly but `suite31_hw` is flat (`|delta| < 0.005`) → tune `max_len` (`512 -> 1536`), `kl_beta`, or reward weights (`r_visual` vs `r_meter`) and test on unlabeled RISM HW pages (`datasets/rism_hw_5707.tar`).
- **DISCARD:** If `suite31_hw` regresses by `> 0.015` across slices or `render_ok_rate < 0.25` (renderer/reward mismatch) → diagnose failure mode in `OUTCOME.md` and pause unlabeled visual RLVR until renderer alignment is fixed.
