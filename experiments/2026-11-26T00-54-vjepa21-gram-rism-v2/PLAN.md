# PLAN — `2026-11-26T00-54-vjepa21-gram-rism-v2`

Zero-Prior-Context Handoff plan for **Cloud Experiment 2**: V-JEPA 2.1 + DINOv3 Gram Anchoring + Stroke-HOG SSL pretraining of `UniOMR v2 small_45m` on 5,707 real RISM historical manuscripts.

---

## 1. Platform Chosen & Rationale

- **Provider:** Modal (`$30/mo` free tier, profile `gregoreesmaa1`)
- **GPU:** `L4` (`24 GB` VRAM, Ada Lovelace `cc=8.9`, **native BF16**, `~$0.80/hr`)
- **Rationale:** `pretrain_ijepa_rism_v2.py` runs dual-encoder (student + EMA teacher + frozen Gram teacher) at `max_dim=1024`, `batch_size=4`, BF16, with native `torch.optim.Muon`. `L4` provides 24 GB VRAM + native BF16 at the lowest Modal $/hr, keeping local RTX 4080 SUPER completely idle.
- **Timeout & Estimated Cost:** `timeout=7200` (2h ceiling). Expected runtime ~20–45 min (`~$0.30–$0.60` of `$30/mo` credit).

---

## 2. Identifiers for Follow-Up Queries

- **Modal Profile:** `gregoreesmaa1`
- **Modal App Name:** `exp2-vjepa21-gram-rism-v2`
- **Modal App ID:** `ap-VDgc7PqnXbaBLqqRVHPNd1`
- **Modal Dashboard URL:** `https://modal.com/apps/gregoreesmaa1/main/ap-VDgc7PqnXbaBLqqRVHPNd1`
- **Modal Shared Volume:** `omr-cache` (mounted at `/vol`)
  - Cached dataset: `/vol/datasets/rism_hw/` (marker: `/vol/datasets/rism_hw/.unpacked_rism_hw_5707`)
  - Checkpoints: `/vol/ckpts/exp2_vjepa21_rism_v2/`
- **Modal Secret:** `hf-secret` (`HF_TOKEN`)
- **Hugging Face Repo:** `gregoreesmaa/sheet-music-converter-omr` (`repo_type="dataset"`)
  - Run ledger: `runs/2026-11-26T00-54-vjepa21-gram-rism-v2.json`
  - Final checkpoint: `checkpoints/2026-11-26T00-54-vjepa21-gram-rism-v2/uni_omr_v2_ijepa_last.pt`
  - Best EMA checkpoint: `checkpoints/2026-11-26T00-54-vjepa21-gram-rism-v2/uni_omr_v2_ijepa_best.pt`
  - Autosave checkpoint (synced every ~20m): `checkpoints/2026-11-26T00-54-vjepa21-gram-rism-v2/uni_omr_v2_ijepa_autosave.pt`
  - Training log: `checkpoints/2026-11-26T00-54-vjepa21-gram-rism-v2/vjepa21_train_log.json`

### Copy-Paste Monitoring / Control Commands (PowerShell)

```powershell
# List active Modal apps
.\.venv\Scripts\modal.exe app list

# Tail live logs
.\.venv\Scripts\modal.exe app logs ap-VDgc7PqnXbaBLqqRVHPNd1

# Stop runaway job if needed
.\.venv\Scripts\modal.exe app stop -y ap-VDgc7PqnXbaBLqqRVHPNd1

# Check HF run ledger & uploaded artifacts
$env:CUDA_VISIBLE_DEVICES=""; .\.venv\Scripts\python.exe -c "from huggingface_hub import HfApi; api=HfApi(); print('\n'.join(f for f in api.list_repo_files('gregoreesmaa/sheet-music-converter-omr', repo_type='dataset') if '2026-11-26T00-54-vjepa21-gram-rism-v2' in f))"
```

---

## 3. Required Training & Eval Data

- **Dataset:** `datasets/rism_hw_5707.tar` on `gregoreesmaa/sheet-music-converter-omr` (`5,707` unlabeled real historical RISM handwritten manuscript pages).
- **License:** Public-domain historical RISM manuscripts (permissive; ZERO `CC-BY-NC`, ZERO `suite31_hw` eval overlap).
- **Initialization:** Fresh `UniOMRv2Net` (`preset="small_45m"`) — baseline commit SHA `9d73170`.

---

## 4. Exact Launch Command & Hyperparameters

### Detached Modal Launch

```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\modal.exe run --detach experiments/2026-11-26T00-54-vjepa21-gram-rism-v2/run_modal.py
```

### Container Subprocess Invocation

```bash
python -u /app/mvd/pretrain_ijepa_rism_v2.py \
  --preset small_45m \
  --image-dir /vol/datasets/rism_hw \
  --out-dir /vol/ckpts/exp2_vjepa21_rism_v2 \
  --steps 1500 \
  --batch-size 4 \
  --grad-accum 2 \
  --max-dim 1024 \
  --bf16 \
  --optimizer muon \
  --gram-weight 1.0 \
  --stroke-hog-weight 0.25 \
  --save-every-minutes 5.0 \
  --save-every 500 \
  --log-every 25
```

- **Effective Batch Size:** `4 × 2 = 8` full pages/step (`12,000` page views over `1,500` steps ≈ `2.1` passes over RISM 5,707).
- **Loss Weights:** DeepStack `(0.25, 0.35, 1.0)`, `masked_weight=3.0`, `visible_weight=0.5`, `gram_weight=1.0` (`gram_max_tokens=512`), `stroke_hog_weight=0.25`, `router_cons_weight=0.2`.
- **Optimizer:** `torch.optim.Muon(adjust_lr_fn="match_rms_adamw")` for 2D encoder Transformer weights + fused `AdamW` (`lr=3e-4`, `weight_decay=0.04`) for backbone/norms/predictors/HOG head.

---

## 5. 5-Minute Checkpointing & Resume Plan

- **Rolling Local Autosave:** `--save-every-minutes 5.0` writes `uni_omr_v2_ijepa_autosave.pt` atomically (`.tmp` + `os.replace`) into `/vol/ckpts/exp2_vjepa21_rism_v2/`. Periodic checkpoints also saved every `500` steps (`uni_omr_v2_ijepa_step{500,1000,1500}.pt`) and on every new best EMA loss (`uni_omr_v2_ijepa_best.pt`).
- **Cloud Persistence:** Background sync thread calls `vol.commit()` every `5 min` and uploads `uni_omr_v2_ijepa_autosave.pt` to HF every `20 min` when updated. Final `uni_omr_v2_ijepa_last.pt`, `uni_omr_v2_ijepa_best.pt`, `vjepa21_train_log.json`, and `runs/2026-11-26T00-54-vjepa21-gram-rism-v2.json` are uploaded on exit.
- **Exact Resume Command (if preempted):**
  ```powershell
  $env:CUDA_VISIBLE_DEVICES=""
  .\.venv\Scripts\modal.exe run --detach experiments/2026-11-26T00-54-vjepa21-gram-rism-v2/run_modal.py `
    --init-from /vol/ckpts/exp2_vjepa21_rism_v2/uni_omr_v2_ijepa_autosave.pt
  ```

---

## 6. Evaluation Gate & Decision Tree ("What We Do With the Result")

Because this is an **encoder-only SSL pretraining run** (decoder/LM head remain frozen), evaluation has two gates:

1. **Immediate SSL Health Gate (at completion of this run):**
   - Zero NaN/Inf collapses, `oom_skips == 0` (or `< 5`), steady monotonic decline in `jepa_masked`, `deep_sup`, and `stroke_hog` while `gram` stays bounded (confirming no dense geometric collapse).
   - `uni_omr_v2_ijepa_last.pt` verified loadable into `UniOMRv2Net(UniOMRv2Config.small_45m())` with 0 missing/unexpected encoder keys.
2. **Downstream Supervised Gate (next experiment — `UniOMR v2 small_45m` fine-tune evaluated on 45-piece `suite31_hw`):**
   - Compare `--init-from uni_omr_v2_ijepa_last.pt` vs random-init `UniOMR v2 small_45m` on `suite31_hw` (`printed_clean`, `printed_photo`, `hw_clean`, `hw_photo`, `all45_clean/photo`).
   - **PROMOTE:** SSL health gate passes AND downstream `hw_clean`/`hw_photo` Strict Note F1 improves by `>= +0.015` without `> 0.005` regression on `printed_clean`/`printed_photo` → make V-JEPA 2.1 RISM pretraining the standard warm-start for `UniOMR v2`.
   - **ITERATE:** SSL loss converges cleanly but downstream gain is `< +0.015` → tune `--gram-weight` / `--steps 4000` / `--max-dim 1280` or initialize from a supervised printed checkpoint before RISM adaptation.
   - **DISCARD:** Representation collapse, NaN instability, or downstream regression on both printed and handwritten splits → revert to scratch / supervised encoder init.
