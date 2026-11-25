# Experiment Plan — `2026-11-26T01-38-ov-39-rism-pseudolabel-threshold-sweep`

## 1. Platform Chosen & Rationale
- **Platform:** **Google Colab `L4` (`24 GB`, `cc=8.9`, native BF16)** via `.\.venv\Scripts\colab.exe`.
- **Rationale:** Native BF16 + MTP-4 speculative FSA decoding runs 300 archival RISM pages in `~12–14 min`, followed by instantaneous 3×3 threshold yield analysis and up to 5 downstream fine-tuning + `suite31_hw` evaluation arms in `~20 min` — fitting cleanly in a single `~33–38 min` (`~2.7–3.0 CU`) L4 run.
- **Estimated Wall-Clock:** `~33–38 min`:
  - Stage 1: FSA-constrained decode + exact `mean_logprob` + `beat_valid_frac` on `300` RISM pages (`max_dim=1024`, `max_len=1024`): `~13 min`
  - Stage 2: 3×3 threshold sweep table (`mean_logprob` × `beat_valid_frac`): `< 5 s`
  - Stage 3: Baseline eval (`~1.5 min`) + up to `5` distinct qualifying threshold arms × (`60` fine-tune steps `~2.5 min` + `52`-page eval `~1.5 min`): `~21.5 min`

## 2. Identifiers for Follow-Up Queries (Zero-Prior-Context Handoff)
- **Experiment ID:** `2026-11-26T01-38-ov-39-rism-pseudolabel-threshold-sweep`
- **Batch / Wave:** `experiments/overnight_queue/batch_09.json` (`wave: 7`, `num: 39`)
- **Colab Session / CLI:** Ephemeral `colab run --gpu L4` or named session `ov-39-rism-pseudo`
- **Hugging Face Repo:** `gregoreesmaa/sheet-music-converter-omr` (`repo_type="dataset"`)
  - Final Eval JSON: `evals/2026-11-26T01-38-ov-39-rism-pseudolabel-threshold-sweep.json`
  - Run Ledger JSON: `runs/2026-11-26T01-38-ov-39-rism-pseudolabel-threshold-sweep.json`

### Copy-Paste Launch & Download Commands
```powershell
# 1. Stage /content/bundle.tar.gz and /tmp/.hf_token (chmod 600) in a Colab L4 session, then run:
.\.venv\Scripts\colab.exe new -s ov-39-rism-pseudo --gpu L4
.\.venv\Scripts\colab.exe upload -s ov-39-rism-pseudo bundle.tar.gz /content/bundle.tar.gz
.\.venv\Scripts\colab.exe upload -s ov-39-rism-pseudo .hf_token_tmp /tmp/.hf_token
.\.venv\Scripts\colab.exe exec -s ov-39-rism-pseudo -f experiments/2026-11-26T01-38-ov-39-rism-pseudolabel-threshold-sweep/run_colab.py
.\.venv\Scripts\colab.exe stop -s ov-39-rism-pseudo

# 2. Download final eval JSON from Hugging Face
.\.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('gregoreesmaa/sheet-music-converter-omr', 'evals/2026-11-26T01-38-ov-39-rism-pseudolabel-threshold-sweep.json', repo_type='dataset'))"
```

## 3. Required Data & Baseline Checkpoint (100% Permissive)
- **Unlabeled RISM Pool:** HF `datasets/rism_hw_5707.tar` (deterministic `seed=42` sample of **300** archival RISM pages, 100% permissive).
- **Eval Dataset:** HF `datasets/suite31_hw_eval.tar` (`suite31/suite31_hw_manifest.json` — all `14` real HW manuscripts + `12` stratified printed pieces, `clean` + `photo` = `52` pages).
- **Baseline Checkpoint:** HF `checkpoints/iter10_soup_baseline/uni_omr_soup.pt`.

## 4. Exact Hyperparameters & Sweep Grid
- **Stage 1 Pseudo-Label Generation:**
  - `n_pages = 300`, `max_dim = 1024`, `max_len = 1024`, `use_grammar = True`, `strict_grammar = False`, MTP-4 speculative decoding, BF16.
  - Per-page quality metrics: `mean_logprob` (exact teacher-forced mean log-probability of generated tokens), `beat_valid_frac` (`r_rhythm` from `rlvr_grpo.compute_unlabeled_structural_reward`), `r_parse`, `r_noloop`, `has_eos`, `n_tokens`.
- **Stage 2 Threshold Grid (`9` cells):**
  - `mean_logprob >= {-0.35, -0.20, -0.10}` × `beat_valid_frac >= {0.70, 0.85, 0.95}` (plus base sanity filters: `r_parse >= 0.5`, `r_noloop == 1.0`, `n_tokens >= 24`).
- **Stage 3 Downstream Fine-Tuning (up to 5 distinct qualifying cells with `n_kept >= 6`):**
  - `ft_steps = 60` optimizer steps (`grad_accum = 4`, effective batch `4`, `lr = 1.5e-5`, cosine decay to `3e-6`, `AdamW`, `weight_decay = 0.01`, `grad_clip = 1.0`, `label_smoothing = 0.02`, mild contrast/brightness jitter for noisy-student regularization).
  - Eval decode: `max_dim = 1024`, `max_len = 1536`, MTP-4 + FSA grammar, BF16, CUTLASS/Flash SDPA (`enable_cudnn_sdp(False)`).

## 5. 5-Minute Checkpointing & Resume Plan
- Atomic JSON checkpointing (`_atomic_write_json` via `.tmp` → `os.replace`) to `/content/cache/evals/2026-11-26T01-38-ov-39-rism-pseudolabel-threshold-sweep_partial.json` every `<= 5.0` minutes during Stage 1 (caching all decoded RISM pseudo-labels) and after every Stage 3 fine-tuning arm.
- Re-running `run_colab.py` automatically skips already-decoded RISM pages and already-evaluated fine-tuning arms.
- `/tmp/.hf_token` is `chmod 600`, read once, unlinked immediately, and unlinked again in `finally:`.

## 6. Evaluation Gate & Decision Tree ("What We Do With the Result")
- **Primary Metrics:** Pseudo-label yield (`n_kept / 300`) across all 9 cells, downstream `suite31_hw` `hw_clean` & `hw_photo` `note_f1_py` lift vs `uni_omr_soup.pt`, and `printed_clean` / `printed_photo` retention.
- **PROMOTE:** If a threshold cell yields `>= 10%` of RISM pages (`>= 30/300`) AND downstream fine-tuning achieves `delta_hw_clean >= +0.015` OR `delta_hw_photo >= +0.015` with `printed_clean` drop `<= 0.010` → scale pseudo-labeling with the winning `(mean_logprob, beat_valid_frac)` gate across all `5,707` RISM pages and mix into Iter 11 SFT.
- **ITERATE:** If yield is `< 10%` at the winning precision threshold, or fine-tuning drifts printed accuracy → add 20% printed replay mixing or 2-view agreement (`generate_rism_pseudolabels.py`) in a follow-up run.
- **DISCARD:** If all qualifying threshold arms regress HW Note F1 → document pseudo-label error rate / confirmation-bias findings in `OUTCOME.md` and pivot to human-verified / synthetic HW augmentation.
