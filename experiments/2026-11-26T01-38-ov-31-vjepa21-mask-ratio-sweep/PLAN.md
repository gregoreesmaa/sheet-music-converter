# PLAN — `2026-11-26T01-38-ov-31-vjepa21-mask-ratio-sweep`

Zero-Prior-Context Handoff plan for **Overnight Experiment 31 (Wave 7)**: V-JEPA 2.1 masking ratio sweep (`0.45`, `0.60`, `0.75`) on unlabeled RISM 5,707 manuscripts.

---

## 1. Platform Chosen & Rationale

- **Provider:** Google Colab via `.\.venv\Scripts\colab.exe` (`google-colab-cli`, `--auth oauth2`)
- **GPU:** `L4` (`24 GB` VRAM, Ada Lovelace `cc=8.9`, **native BF16**, `~4.8 CU/hr`)
- **Rationale:** Each 500-step arm at `batch_size=4`, `grad_accum=2`, `max_dim=1024` takes ~10–12 min on `L4` BF16; running all 3 arms + held-out linear/kNN staff-density probes fits cleanly inside a single ~35–40 min ephemeral `L4` session (~3.2 CU) while keeping the local RTX 4080 SUPER free.
- **Estimated Wall-Clock & Cost:** ~35–40 min (`~3.2 CU` out of `1,479.59 CU` balance).

---

## 2. Identifiers for Follow-Up Queries

- **Experiment ID:** `2026-11-26T01-38-ov-31-vjepa21-mask-ratio-sweep`
- **Queue File:** `experiments/overnight_queue/batch_07.json` (`wave: 7`, `status: "QUEUED"`)
- **Colab Session:** Ephemeral `colab run --gpu L4` (or named session `ov-31-vjepa-mask`)
- **Hugging Face Repo:** `gregoreesmaa/sheet-music-converter-omr` (`repo_type="dataset"`)
  - Eval JSON: `evals/2026-11-26T01-38-ov-31-vjepa21-mask-ratio-sweep.json`
  - Run Ledger: `runs/2026-11-26T01-38-ov-31-vjepa21-mask-ratio-sweep.json`

### Copy-Paste Monitoring / Verification Commands (PowerShell)

```powershell
# Check Colab balance and active sessions
.\.venv\Scripts\colab.exe usage
.\.venv\Scripts\colab.exe sessions

# Verify uploaded HF eval & run JSON
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe -c "from huggingface_hub import HfApi; api=HfApi(); print('\n'.join(f for f in api.list_repo_files('gregoreesmaa/sheet-music-converter-omr', repo_type='dataset') if 'ov-31-vjepa21-mask-ratio-sweep' in f))"
```

---

## 3. Required Training & Eval Data

- **Dataset:** `datasets/rism_hw_5707.tar` on `gregoreesmaa/sheet-music-converter-omr` (`5,707` unlabeled real historical RISM handwritten manuscript pages).
- **Deterministic Split (seed `42`):**
  - Training pool: `5,451` pages
  - Held-out probe validation pool: `256` pages (`200` probe-fit / `56` probe-test for Ridge & `k=5` cosine kNN staff-density probes)
- **License:** Public-domain historical RISM manuscripts (permissive; ZERO `CC-BY-NC`, ZERO `suite31_hw` eval overlap).
- **Initialization:** Fresh `UniOMRv2Net(UniOMRv2Config.small_45m())` per arm (identical `seed=42`, baseline SHA `164852f`).

---

## 4. Exact Launch Command & Hyperparameters

```powershell
# Named session workflow (upload repo bundle + /tmp/.hf_token, run, stop)
.\.venv\Scripts\colab.exe new -s ov-31-mask --gpu L4
.\.venv\Scripts\colab.exe upload -s ov-31-mask local_bundle.tar.gz /content/repo_bundle.tar.gz
# Upload HF token to /tmp/.hf_token (deleted automatically by run_colab.py in finally)
.\.venv\Scripts\colab.exe exec -s ov-31-mask -f experiments/2026-11-26T01-38-ov-31-vjepa21-mask-ratio-sweep/run_colab.py
.\.venv\Scripts\colab.exe stop -s ov-31-mask
```

### Controlled Sweep Configuration

- **Single Swept Variable:** `mask_ratio` (`mask_ratio_range = (r - 0.05, r + 0.05)` passed to `apply_ink_span_mask_and_corruption`):
  1. `mask_045`: `r = 0.45` (`[0.40, 0.50]`)
  2. `mask_060`: `r = 0.60` (`[0.55, 0.65]`)
  3. `mask_075`: `r = 0.75` (`[0.70, 0.80]`)
- **Fixed Across All Arms:**
  - `preset = "small_45m"`, `steps = 500`, `batch_size = 4`, `grad_accum = 2` (eff. batch 8), `max_dim = 1024`, `bf16 = True`
  - `optimizer = "muon"` (`Muon(adjust_lr_fn="match_rms_adamw")` + fused `AdamW`, `lr = 3e-4`, `weight_decay = 0.04`, `warmup_steps = 50`)
  - `level_weights = (0.25, 0.35, 1.0)`, `masked_weight = 3.0`, `visible_weight = 0.5`, `gram_weight = 1.0`, `gram_max_tokens = 512`, `stroke_hog_weight = 0.25`, `router_cons_weight = 0.2`
  - `ema_start = 0.996`, `ema_end = 0.999`, `seed = 42`
  - CUTLASS `EFFICIENT_ATTENTION` enforced (`enable_cudnn_sdp(False)`).

---

## 5. 5-Minute Checkpointing & Resume Plan

- Every arm saves atomic `.tmp` → `os.replace` rolling checkpoints (`uni_omr_v2_ijepa_autosave.pt`) every **`5.0` minutes** into `/content/vol/ckpts/ov31/<arm>/`, plus `uni_omr_v2_ijepa_last.pt` on arm completion.
- Completed arm metrics are checkpointed atomically to `/content/vol/evals/ov31_progress.json` after each arm so a preempted run can resume without re-running finished arms.
- `/tmp/.hf_token` is read once into memory and deleted immediately + guaranteed deleted in `finally:`.

---

## 6. Evaluation Gate & Decision Tree ("What We Do With the Result")

- **Primary Metrics (per arm on 256 held-out RISM pages):**
  1. **Linear (Ridge) Staff-Density Probe `R^2` & `MAE`** predicting page ink/horizontal-staff density from frozen EMA teacher pooled embeddings.
  2. **`k=5` Cosine kNN Staff-Density Probe `R^2` & Quartile Accuracy**.
  3. **Loss Stability:** `final_ema_loss`, tail-100 loss std, `nan_skips`, `oom_skips`, realized `masked_ink_frac`.
- **PROMOTE:** Winning `mask_ratio` beats the `0.45` baseline by `>= +0.03` probe `R^2` (Ridge or kNN) with `nan_skips == 0` and stable tail loss → add `--mask-ratio` CLI flag to `tmp-train/mvd/pretrain_ijepa_rism_v2.py` and set the winner as default for Iter 11+ RISM SSL.
- **ITERATE:** All arms are within `< 0.03` probe `R^2` or `0.75` improves probe `R^2` with mild variance → test intermediate `0.65` or scheduled curriculum masking (`0.45 -> 0.65`).
- **DISCARD:** High masking (`0.60`/`0.75`) degrades probe `R^2` or causes NaN/loss spikes → keep the baseline `(0.45, 0.55)` default in `pretrain_ijepa_rism_v2.py`.
