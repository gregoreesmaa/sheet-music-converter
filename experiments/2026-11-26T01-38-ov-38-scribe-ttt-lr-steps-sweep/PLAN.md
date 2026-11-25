# Experiment Plan — `2026-11-26T01-38-ov-38-scribe-ttt-lr-steps-sweep`

## 1. Platform Chosen & Rationale
- **Platform:** **Google Colab `L4` (`24 GB`, `cc=8.9`, native BF16)** via `.\.venv\Scripts\colab.exe`.
- **Rationale:** Native BF16 + pre-installed `torch 2.14.1+cu130` (~8s cold start), 24 GB VRAM comfortably supports full-encoder autograd at `1280px` with CPU-offloaded bit-exact parameter snapshots, and cumulative stepping (`0 → 2 → 4 → 8`) + shared pre-TTT pseudo-decode keeps all 18 grid cells across 28 pages within a single ~26–32 min run (`~2.2–2.6 CU`).
- **Estimated Wall-Clock:** `~26–32 min`:
  - `28` pages × (`1` baseline decode + `1` cached pre-TTT pseudo-decode + `6` `(scope, lr)` cumulative trajectories × `[8 TTT steps + 3 MTP-4 decodes]`): `~50–60 s/page` ≈ `~25–28 min` + `~2 min` asset setup.

## 2. Identifiers for Follow-Up Queries (Zero-Prior-Context Handoff)
- **Experiment ID:** `2026-11-26T01-38-ov-38-scribe-ttt-lr-steps-sweep`
- **Batch / Wave:** `experiments/overnight_queue/batch_09.json` (`wave: 7`, `num: 38`)
- **Colab Session / CLI:** Ephemeral `colab run --gpu L4` or named session `ov-38-scribe-ttt`
- **Hugging Face Repo:** `gregoreesmaa/sheet-music-converter-omr` (`repo_type="dataset"`)
  - Final Eval JSON: `evals/2026-11-26T01-38-ov-38-scribe-ttt-lr-steps-sweep.json`
  - Run Ledger JSON: `runs/2026-11-26T01-38-ov-38-scribe-ttt-lr-steps-sweep.json`

### Copy-Paste Launch & Download Commands
```powershell
# 1. Stage /content/bundle.tar.gz and /tmp/.hf_token (chmod 600) in a Colab L4 session, then run:
.\.venv\Scripts\colab.exe new -s ov-38-scribe-ttt --gpu L4
.\.venv\Scripts\colab.exe upload -s ov-38-scribe-ttt bundle.tar.gz /content/bundle.tar.gz
.\.venv\Scripts\colab.exe upload -s ov-38-scribe-ttt .hf_token_tmp /tmp/.hf_token
.\.venv\Scripts\colab.exe exec -s ov-38-scribe-ttt -f experiments/2026-11-26T01-38-ov-38-scribe-ttt-lr-steps-sweep/run_colab.py
.\.venv\Scripts\colab.exe stop -s ov-38-scribe-ttt

# 2. Download final eval JSON from Hugging Face
.\.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('gregoreesmaa/sheet-music-converter-omr', 'evals/2026-11-26T01-38-ov-38-scribe-ttt-lr-steps-sweep.json', repo_type='dataset'))"
```

## 3. Required Data & Baseline Checkpoint (100% Permissive)
- **Eval Dataset:** HF `datasets/suite31_hw_eval.tar` (`suite31/suite31_hw_manifest.json`) — all **14 real historical handwritten manuscripts** across `clean` and `photo` (`28` pages total, 100% permissive).
- **Baseline Checkpoint:** HF `checkpoints/iter10_soup_baseline/uni_omr_soup.pt`.

## 4. Exact Hyperparameters & Sweep Grid
- **Grid (`18` cells):**
  - `adapt_scope` ∈ `{"encoder_norm_only", "full_encoder"}`
    - `encoder_norm_only`: `RMSNorm` scale weights in `backbone`, `enc_layers`, `enc_norm` (`< 0.1%` params)
    - `full_encoder`: all parameters in `backbone`, `router`, `enc_layers`, `enc_norm` (CPU snapshot, bit-exact `atol=0` restore)
  - `ttt_lr` ∈ `{1e-5, 3e-5, 1e-4}`
  - `steps` ∈ `{2, 4, 8}` (evaluated cumulatively within each `(scope, lr)` AdamW trajectory)
- **Fixed SCRIBE/DocTTT Controls:**
  - `conf_threshold = 0.85`, `cons_weight = 1.0`, `ent_weight = 1.0`, `kl_weight = 0.5`, `l2_weight = 1e-2`, `grad_clip = 1.0`
  - `max_dim = 1280` (OOM fallback `1024`), `max_len = 1536`, `ttt_max_decode_len = 512`, MTP-4 speculative + FSA grammar (`strict=False`), BF16, CUTLASS/Flash SDPA (`enable_cudnn_sdp(False)`).

## 5. 5-Minute Checkpointing & Resume Plan
- Atomic JSON progress checkpointing (`_atomic_write_json` via `.tmp` → `os.replace`) to `/content/cache/evals/2026-11-26T01-38-ov-38-scribe-ttt-lr-steps-sweep_partial.json` every `<= 5.0` minutes and after every completed `(piece, domain)` page.
- Re-running `run_colab.py` automatically skips completed `(piece, domain)` pairs.
- `/tmp/.hf_token` is `chmod 600`, read once, unlinked immediately, and unlinked again in `finally:`.

## 6. Evaluation Gate & Decision Tree ("What We Do With the Result")
- **Primary Metrics:** `hw_clean` and `hw_photo` `note_f1_py` delta vs pre-TTT baseline, win/tie/loss ratio across the 14 HW manuscripts, structural reward (`r_rhythm`, `r_noloop`), and median adaptation wall-clock seconds/page across all 18 cells.
- **PROMOTE:** If any `(scope, lr, steps)` cell achieves `delta_note_f1_py >= +0.020` on `hw_clean` OR `hw_photo` with `wins > losses` and no regression `> -0.005` on the other domain → wire the winning `(scope, lr, steps)` as the default `--ttt-mode` preset in `trial_2028_ttt_adapt.py` and verify zero regression on printed pieces.
- **ITERATE:** If `full_encoder` beats `encoder_norm_only` at `steps=2..4` but regresses at `steps=8`, or gains are concentrated on high-`n_confident` pages → add a confidence-count gate (`n_confident >= 32`) before applying TTT updates.
- **DISCARD:** If all 18 cells yield `delta_note_f1_py <= +0.003` or regress → record failure mode in `OUTCOME.md` and defer TTT until base HW Note F1 exceeds `0.40+`.
