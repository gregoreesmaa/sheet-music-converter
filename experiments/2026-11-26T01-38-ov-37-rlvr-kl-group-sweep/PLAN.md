# Experiment Plan — `2026-11-26T01-38-ov-37-rlvr-kl-group-sweep`

## 1. Platform Chosen & Rationale
- **Platform:** **Google Colab `L4` (`24 GB`, `cc=8.9`, native BF16)** via `.\.venv\Scripts\colab.exe`.
- **Rationale:** Native BF16 + pre-installed `torch 2.14.1+cu130` (~8s cold start, ~4.8 CU/hr), zero contention with local RTX 4080 SUPER, and 24 GB VRAM comfortably holds `UniOMRNet` + frozen `ref_model` + `G=8` rollouts (`max_len=256`, `max_dim=1024`).
- **Estimated Wall-Clock:** `~32–38 min` (`~2.7–3.0 CU`):
  - `1` baseline eval (`14` HW + `12` stratified Printed = `52` pages): `~1.5 min`
  - `6` RLVR arms × `40` GRPO updates (`3 × G=4` @ `~2.5 min` + `3 × G=8` @ `~4.5 min`): `~21 min`
  - `6` post-arm evals (`52` pages each): `~9 min`
  - `1` full 45-piece `suite31_hw` (`90` pages) confirmation eval on the winning arm: `~2.5 min`

## 2. Identifiers for Follow-Up Queries (Zero-Prior-Context Handoff)
- **Experiment ID:** `2026-11-26T01-38-ov-37-rlvr-kl-group-sweep`
- **Batch / Wave:** `experiments/overnight_queue/batch_09.json` (`wave: 7`, `num: 37`)
- **Colab Session / CLI:** Ephemeral `colab run --gpu L4` or named session `ov-37-rlvr`
- **Hugging Face Repo:** `gregoreesmaa/sheet-music-converter-omr` (`repo_type="dataset"`)
  - Final Eval JSON: `evals/2026-11-26T01-38-ov-37-rlvr-kl-group-sweep.json`
  - Run Ledger JSON: `runs/2026-11-26T01-38-ov-37-rlvr-kl-group-sweep.json`
  - Winning Arm Checkpoint (only if Promote gate passes): `checkpoints/2026-11-26T01-38-ov-37-rlvr-kl-group-sweep/best_rlvr_arm.pt`

### Copy-Paste Launch & Download Commands
```powershell
# 1. Stage /content/bundle.tar.gz and /tmp/.hf_token (chmod 600) in a Colab L4 session, then run:
.\.venv\Scripts\colab.exe new -s ov-37-rlvr --gpu L4
.\.venv\Scripts\colab.exe upload -s ov-37-rlvr bundle.tar.gz /content/bundle.tar.gz
.\.venv\Scripts\colab.exe upload -s ov-37-rlvr .hf_token_tmp /tmp/.hf_token
.\.venv\Scripts\colab.exe exec -s ov-37-rlvr -f experiments/2026-11-26T01-38-ov-37-rlvr-kl-group-sweep/run_colab.py
.\.venv\Scripts\colab.exe stop -s ov-37-rlvr

# 2. Download final eval JSON from Hugging Face
.\.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('gregoreesmaa/sheet-music-converter-omr', 'evals/2026-11-26T01-38-ov-37-rlvr-kl-group-sweep.json', repo_type='dataset'))"
```

## 3. Required Data & Baseline Checkpoint (100% Permissive)
- **Unlabeled RLVR Corpus:** HF `datasets/rism_hw_5707.tar` (`5,707` unlabeled archival RISM handwritten pages, 100% permissive; zero `CC-BY-NC`).
- **Eval Dataset:** HF `datasets/suite31_hw_eval.tar` (`suite31/suite31_hw_manifest.json` — all `14` real historical HW manuscripts + `12` stratified printed pieces for sweep, and all `45` pieces = `90` pages for final winner confirmation).
- **Baseline Checkpoint:** HF `checkpoints/iter10_soup_baseline/uni_omr_soup.pt`.

## 4. Exact Hyperparameters & Sweep Grid
- **Grid (`6` arms):** `kl_beta` ∈ `{0.02, 0.06, 0.15}` × `group_size` ∈ `{4, 8}`.
- **Fixed RLVR Controls (matched across all 6 arms):**
  - `updates = 40` (same deterministic shuffled RISM page order `seed=42` across all arms for paired comparison)
  - `lr = 1.0e-5` (4-step linear warmup + cosine decay to `2.0e-6`, `AdamW`, `weight_decay = 0.01`, `grad_clip = 1.0`)
  - `max_len = 256`, `max_dim = 1024`, `temperature = 0.8`, `top_p = 0.95`, `clip_eps = 0.2`
  - `use_grammar = True`, `strict_grammar = False`, `unlabeled_frac = 1.0` (`compute_unlabeled_structural_reward`: parse + beat-conservation `r_rhythm` + `<sys>`/EOS completeness + anti-loop)
  - Eval decode: `max_dim = 1024`, `max_len = 1536`, MTP-4 speculative + FSA grammar, BF16, CUTLASS/Flash SDPA (`enable_cudnn_sdp(False)`).

## 5. 5-Minute Checkpointing & Resume Plan
- Atomic JSON progress checkpointing (`_atomic_write_json` via `.tmp` → `os.replace`) to `/content/cache/evals/2026-11-26T01-38-ov-37-rlvr-kl-group-sweep_partial.json` every `<= 5.0` minutes and after every completed arm.
- Re-running `run_colab.py` in the same session automatically loads cached baseline + completed arms and resumes remaining arms.
- `/tmp/.hf_token` is `chmod 600`, read into memory, unlinked immediately, and unlinked again in `finally:`.

## 6. Evaluation Gate & Decision Tree ("What We Do With the Result")
- **Primary Metrics:** `hw_clean` & `hw_photo` `note_f1_py`, `r_rhythm` (beat conservation), `r_noloop`, and `printed_clean` / `printed_photo` retention deltas vs `uni_omr_soup.pt`.
- **PROMOTE:** If any `(kl_beta, G)` arm achieves `delta_hw_clean >= +0.015` OR `delta_hw_photo >= +0.015` with `printed_clean` regression `<= 0.010` and `r_rhythm` improving → promote winning `(kl_beta, G)` to full 250-update RLVR post-training on Iter 11.
- **ITERATE:** If `r_rhythm` and `r_noloop` improve significantly (`>= +0.05`) without Note F1 gain, or if `G=8` clearly beats `G=4` only at high `kl_beta` → test mixed labeled+unlabeled RLVR (`unlabeled_frac=0.50`) with the winning `(kl_beta, G)`.
- **DISCARD:** If all 6 arms regress HW Note F1 or degrade printed Note F1 by `> 0.015` → document reward-hacking / KL-collapse diagnostics in `OUTCOME.md` and gate RLVR behind a visual render-diff verifier.
