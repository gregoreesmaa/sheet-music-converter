# PLAN — `2026-11-26T01-38-ov-33-vjepa-warmstart-vs-scratch`

Zero-Prior-Context Handoff plan for **Overnight Experiment 33 (Wave 7)**: Controlled downstream OMR transfer probe comparing V-JEPA 2.1 RISM-pretrained encoder initialization vs random scratch initialization on `iter10_5_proxy_2k` (800 steps) + `suite31_hw` OOD evaluation.

---

## 1. Platform Chosen & Rationale

- **Provider:** Google Colab via `.\.venv\Scripts\colab.exe` (`google-colab-cli`, `--auth oauth2`)
- **GPU:** `L4` (`24 GB` VRAM, Ada Lovelace `cc=8.9`, **native BF16**, `~4.8 CU/hr`)
- **Rationale:** Two 800-step `UniOMR v2 small_45m` proxy runs (`1280px`, `BF16`, axial-only `--grad-checkpoint`) plus `suite31_hw` evaluation fit comfortably in `24 GB` VRAM and complete in ~35–45 min (~3.6 CU) on `L4`, keeping the local RTX 4080 SUPER completely idle.
- **Estimated Wall-Clock & Cost:** ~35 min if Exp 2 HF checkpoint is already uploaded; ~45 min if the 400-step RISM warmup fallback runs (`~3.0–3.6 CU`).

---

## 2. Identifiers for Follow-Up Queries

- **Experiment ID:** `2026-11-26T01-38-ov-33-vjepa-warmstart-vs-scratch`
- **Queue File:** `experiments/overnight_queue/batch_07.json` (`wave: 7`, `status: "QUEUED"`)
- **Colab Session:** Ephemeral `colab run --gpu L4` (or named session `ov-33-transfer`)
- **Hugging Face Repo:** `gregoreesmaa/sheet-music-converter-omr` (`repo_type="dataset"`)
  - Upstream SSL Checkpoint (preferred if present): `checkpoints/2026-11-26T00-54-vjepa21-gram-rism-v2/uni_omr_v2_ijepa_best.pt` (fallback: `uni_omr_v2_ijepa_last.pt`, else 400-step RISM warmup)
  - Eval JSON: `evals/2026-11-26T01-38-ov-33-vjepa-warmstart-vs-scratch.json`
  - Run Ledger: `runs/2026-11-26T01-38-ov-33-vjepa-warmstart-vs-scratch.json`

### Copy-Paste Monitoring / Verification Commands (PowerShell)

```powershell
.\.venv\Scripts\colab.exe usage
.\.venv\Scripts\colab.exe sessions

$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe -c "from huggingface_hub import HfApi; api=HfApi(); print('\n'.join(f for f in api.list_repo_files('gregoreesmaa/sheet-music-converter-omr', repo_type='dataset') if 'ov-33-vjepa-warmstart-vs-scratch' in f))"
```

---

## 3. Required Training & Eval Data

- **Supervised Proxy Dataset:** `datasets/iter10_5_proxy_2k.tar` (`~2,000` clean+photo full-page OMR samples, permissive license, zero `CC-BY-NC`, zero `suite31_hw` overlap).
- **Optional SSL Warmup Dataset (only if Exp 2 checkpoint not yet on HF):** `datasets/rism_hw_5707.tar` (`5,707` unlabeled public-domain RISM historical manuscripts).
- **OOD Evaluation Dataset:** `datasets/suite31_hw_eval.tar` (`suite31/suite31_hw_manifest.json` — 45 pieces: 31 printed + 14 real historical handwritten manuscripts, `clean` + `photo` = 90 pages).
- **Baseline Commit SHA:** `164852f`.

---

## 4. Exact Launch Command & Hyperparameters

```powershell
.\.venv\Scripts\colab.exe new -s ov-33-transfer --gpu L4
.\.venv\Scripts\colab.exe upload -s ov-33-transfer local_bundle.tar.gz /content/repo_bundle.tar.gz
# Upload HF token to /tmp/.hf_token (deleted automatically by run_colab.py in finally)
.\.venv\Scripts\colab.exe exec -s ov-33-transfer -f experiments/2026-11-26T01-38-ov-33-vjepa-warmstart-vs-scratch/run_colab.py
.\.venv\Scripts\colab.exe stop -s ov-33-transfer
```

### Controlled 2-Arm Configuration (`tmp-train/mvd/train_uni_v2.py`)

- **Single Swept Variable:** Encoder initialization (`--init-encoder-from` omitted for `arm_a_scratch` vs V-JEPA 2.1 RISM checkpoint for `arm_b_vjepa_warmstart`).
- **Fixed Across Both Arms:**
  - `--preset small_45m --curriculum fullpage --steps 800 --batch-size 2 --grad-accum 4`
  - `--fullpage-max-dim 1280 --max-seq-len 1536 --bf16 --grad-checkpoint`
  - `--optimizer muon --lr 3e-4 --weight-decay 0.01 --lr-schedule wsd --warmup-steps 80`
  - `--eval-every 400 --save-every-minutes 5.0 --log-every 25 --seed 42`
- **OOD Evaluation (`tmp-train/mvd/eval_uni_suite31.py`):**
  - `--max-dim 1280 --max-len 768 --bf16` on `suite31_hw_manifest.json` (both arms evaluated identically with Linux Rust `assembly-serialize` + `eval-harness` when built).

---

## 5. 5-Minute Checkpointing & Resume Plan

- Both arms pass `--save-every-minutes 5.0` to `train_uni_v2.py` (atomic `.tmp` → `os.replace` for `uni_omr_v2_latest.pt` and `uni_omr_v2_ema_latest.pt`) under `/content/vol/ckpts/ov33/<arm>/`.
- If the 400-step RISM fallback runs, it also passes `--save-every-minutes 5.0` under `/content/vol/ckpts/ov33/rism_warmup/`.
- `/tmp/.hf_token` is deleted immediately after read and guaranteed deleted in `finally:`.

---

## 6. Evaluation Gate & Decision Tree ("What We Do With the Result")

- **Primary Metrics (`suite31_hw` OOD, Arm B minus Arm A):**
  - `hw_clean` & `hw_photo` Strict Note F1 (and Token F1)
  - `printed_clean` & `printed_photo` Strict Note F1 (guardrail against printed regression)
  - `all45_clean` & `all45_photo` Strict Note F1, plus held-out `proxy_val` SER.
- **PROMOTE:** `arm_b_vjepa_warmstart` improves `hw_clean` or `hw_photo` Strict Note F1 (or Token F1 if Rust scorer absent) by `>= +0.015` with `<= 0.005` regression on `printed_clean`/`printed_photo` → make RISM V-JEPA 2.1 `--init-encoder-from` mandatory for full Iter 11 `UniOMR v2` training.
- **ITERATE:** Gain is positive (`+0.005` to `+0.015`) or printed slightly regresses → combine winning `mask_ratio` (Exp 31) and `gram_weight` (Exp 32) with lower encoder LR (`0.3×` layerwise decay) during supervised fine-tuning.
- **DISCARD:** Warm-start matches or underperforms scratch across both `hw_*` and `printed_*` splits → investigate encoder-decoder alignment / feature scaling before investing further in SSL warm-starts.
