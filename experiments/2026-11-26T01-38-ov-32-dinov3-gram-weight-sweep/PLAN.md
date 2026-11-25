# PLAN — `2026-11-26T01-38-ov-32-dinov3-gram-weight-sweep`

Zero-Prior-Context Handoff plan for **Overnight Experiment 32 (Wave 7)**: DINOv3 Ink-Token Gram Anchoring weight sweep (`0.0`, `0.1`, `0.5`) on unlabeled RISM 5,707 manuscripts, measuring dense patch-feature rank and spectral entropy to prevent feature collapse.

---

## 1. Platform Chosen & Rationale

- **Provider:** Google Colab via `.\.venv\Scripts\colab.exe` (`google-colab-cli`, `--auth oauth2`)
- **GPU:** `L4` (`24 GB` VRAM, Ada Lovelace `cc=8.9`, **native BF16**, `~4.8 CU/hr`)
- **Rationale:** 3 arms × 500 steps (`batch_size=4`, `grad_accum=2`, `max_dim=1024`, BF16) + dense SVD/Gram feature-collapse diagnostics on 128 held-out RISM pages completes in ~35–40 min (~3.2 CU) on `L4`, leaving local RTX 4080 SUPER idle.
- **Estimated Wall-Clock & Cost:** ~35–40 min (`~3.2 CU`).

---

## 2. Identifiers for Follow-Up Queries

- **Experiment ID:** `2026-11-26T01-38-ov-32-dinov3-gram-weight-sweep`
- **Queue File:** `experiments/overnight_queue/batch_07.json` (`wave: 7`, `status: "QUEUED"`)
- **Colab Session:** Ephemeral `colab run --gpu L4` (or named session `ov-32-gram`)
- **Hugging Face Repo:** `gregoreesmaa/sheet-music-converter-omr` (`repo_type="dataset"`)
  - Eval JSON: `evals/2026-11-26T01-38-ov-32-dinov3-gram-weight-sweep.json`
  - Run Ledger: `runs/2026-11-26T01-38-ov-32-dinov3-gram-weight-sweep.json`

### Copy-Paste Monitoring / Verification Commands (PowerShell)

```powershell
.\.venv\Scripts\colab.exe usage
.\.venv\Scripts\colab.exe sessions

$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe -c "from huggingface_hub import HfApi; api=HfApi(); print('\n'.join(f for f in api.list_repo_files('gregoreesmaa/sheet-music-converter-omr', repo_type='dataset') if 'ov-32-dinov3-gram-weight-sweep' in f))"
```

---

## 3. Required Training & Eval Data

- **Dataset:** `datasets/rism_hw_5707.tar` on `gregoreesmaa/sheet-music-converter-omr` (`5,707` unlabeled real historical RISM handwritten manuscript pages).
- **Deterministic Split (seed `42`):**
  - Training pool: `5,579` pages
  - Held-out dense-feature diagnostic pool: `128` pages
- **License:** Public-domain historical RISM manuscripts (permissive; ZERO `CC-BY-NC`, ZERO `suite31_hw` eval overlap).
- **Initialization:** Fresh `UniOMRv2Net(UniOMRv2Config.small_45m())` per arm (identical `seed=42`, baseline SHA `164852f`).

---

## 4. Exact Launch Command & Hyperparameters

```powershell
.\.venv\Scripts\colab.exe new -s ov-32-gram --gpu L4
.\.venv\Scripts\colab.exe upload -s ov-32-gram local_bundle.tar.gz /content/repo_bundle.tar.gz
# Upload HF token to /tmp/.hf_token (deleted automatically by run_colab.py in finally)
.\.venv\Scripts\colab.exe exec -s ov-32-gram -f experiments/2026-11-26T01-38-ov-32-dinov3-gram-weight-sweep/run_colab.py
.\.venv\Scripts\colab.exe stop -s ov-32-gram
```

### Controlled Sweep Configuration

- **Single Swept Variable:** `gram_weight`:
  1. `gram_0p0`: `gram_weight = 0.0` (no Gram anchoring)
  2. `gram_0p1`: `gram_weight = 0.1`
  3. `gram_0p5`: `gram_weight = 0.5`
- **Fixed Across All Arms:**
  - `preset = "small_45m"`, `steps = 500`, `batch_size = 4`, `grad_accum = 2`, `max_dim = 1024`, `bf16 = True`
  - `optimizer = "muon"` (`lr = 3e-4`, `weight_decay = 0.04`, `warmup_steps = 50`)
  - `mask_ratio_range = (0.45, 0.55)`, `level_weights = (0.25, 0.35, 1.0)`, `masked_weight = 3.0`, `visible_weight = 0.5`, `gram_max_tokens = 512`, `stroke_hog_weight = 0.25`, `router_cons_weight = 0.2`, `seed = 42`
  - CUTLASS `EFFICIENT_ATTENTION` enforced (`enable_cudnn_sdp(False)`).

---

## 5. 5-Minute Checkpointing & Resume Plan

- Atomic `.tmp` → `os.replace` rolling checkpoints saved every **`5.0` minutes** into `/content/vol/ckpts/ov32/<arm>/uni_omr_v2_ijepa_autosave.pt`, plus `uni_omr_v2_ijepa_last.pt` on arm completion.
- Per-arm diagnostics written atomically to `/content/vol/evals/ov32_progress.json` after each arm.
- `/tmp/.hf_token` deleted immediately after read and guaranteed deleted in `finally:`.

---

## 6. Evaluation Gate & Decision Tree ("What We Do With the Result")

- **Primary Dense Feature-Collapse Metrics (on 128 held-out RISM pages, dense `mem_final` kept ink tokens `F in R^{K x D}`):**
  1. **Effective Rank (`erank`):** `exp(H(sigma_i / sum(sigma)))` (higher = richer non-collapsed feature space; collapse warning if `erank < 16`).
  2. **Normalized Spectral Entropy (`H_norm in [0, 1]`):** `H(p) / log(min(K, D))` and **Top-1 Variance Ratio** `sigma_1^2 / sum(sigma_i^2)`.
  3. **Mean Off-Diagonal Pairwise Cosine Similarity:** average `cos(f_i, f_j)` (`i != j`) across subsampled ink tokens (lower = better spatial token diversity; collapse if `> 0.85`).
  4. **Gram Drift vs Step-0 Frozen Teacher:** `MSE(G_final, G_step0)` — verifies the encoder actually adapts rather than staying frozen.
  5. **SSL Loss & Linear Staff-Density Probe `R^2`**.
- **PROMOTE:** Winning `gram_weight > 0` increases `erank` by `>= +15%` (or cuts off-diagonal cosine by `>= 0.05`) vs `gram_weight=0.0` while maintaining `jepa_masked` convergence and non-trivial Gram drift → set as the default `--gram-weight` in `pretrain_ijepa_rism_v2.py`.
- **ITERATE:** Both `0.1` and `0.5` prevent collapse equally, or `0.5` over-constrains `jepa_masked` → test `0.05` vs `0.25` or cosine-decayed Gram weight.
- **DISCARD:** `gram_weight=0.0` shows high `erank` (`> 64`) and equal/better probe `R^2` with zero collapse → disable Gram teacher to save ~20% compute/VRAM.
