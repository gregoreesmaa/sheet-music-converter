# Plan — `2026-11-26T01-38-ov-17-irope-registers-jitter-sweep`

## 1. Platform Chosen & Rationale

- **Platform:** Google Colab `L4` (24 GB VRAM, `cc=8.9`, native BF16, `torch 2.14.1+cu130` pre-installed).
- **Rationale:** 6 controlled short proxy arms (`300` steps each at `1280px`) run cleanly in ~38–42 min (~3.4 CU) on a single L4 with identical seed, data order, and optimizer state.

---

## 2. Identifiers for Follow-Up Queries

- **Experiment ID:** `2026-11-26T01-38-ov-17-irope-registers-jitter-sweep`
- **Colab Session Name (recommended):** `ov-17-irope-reg`
- **Colab Commands:**
  - Status: `.\.venv\Scripts\colab.exe sessions`
  - Usage: `.\.venv\Scripts\colab.exe usage`
  - Stop (MANDATORY on finish): `.\.venv\Scripts\colab.exe stop -s ov-17-irope-reg`
- **Hugging Face Private Dataset Hub (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-26T01-38-ov-17-irope-registers-jitter-sweep.json`
  - Run JSON: `runs/2026-11-26T01-38-ov-17-irope-registers-jitter-sweep.json`

---

## 3. Required Training & Eval Data

- **Dataset Tar:** `datasets/iter10_5_proxy_2k.tar` on `gregoreesmaa/sheet-music-converter-omr` (permissive license only; zero `CC-BY-NC`, zero `suite31_hw` eval overlap).
  - `proxy_train.jsonl`: 2,000 full-page training samples.
  - `proxy_val.jsonl`: 64 held-out full-page validation samples.
- **Model Init:** Fresh `UniOMRv2Config.small_45m()` (`seed=42`) with `n_register_tokens` and `rope_jitter` injected cleanly via `train_uni_v2.build_v2_config` override per arm (no edits to core training script required).

---

## 4. Exact Launch Command & Hyperparameters

```powershell
# 1. Provision L4 session & upload code bundle + HF token (zero secrets in git)
.\.venv\Scripts\colab.exe new -s ov-17-irope-reg --gpu L4
.\.venv\Scripts\colab.exe upload -s ov-17-irope-reg /path/to/mvd_bundle.tar.gz /content/bundle.tar.gz
.\.venv\Scripts\colab.exe upload -s ov-17-irope-reg /path/to/local_hf_token /tmp/.hf_token

# 2. Execute standalone experiment runner
.\.venv\Scripts\colab.exe exec -s ov-17-irope-reg -f experiments/2026-11-26T01-38-ov-17-irope-registers-jitter-sweep/run_colab.py

# 3. Tear down immediately after completion (zero idle CU burn)
.\.venv\Scripts\colab.exe stop -s ov-17-irope-reg
```

### Matched Hyperparameters Across All 6 Arms (`train_uni_v2.py`)

- `--preset small_45m`, `irope_nope_every=4`
- `--optimizer muon`, `--lr 3e-4`, `--weight-decay 0.01`, `--lr-schedule wsd`, `--warmup-steps 45`, `--wsd-decay-frac 0.25`, `--min-lr-ratio 0.05`
- `--steps 300`, `--batch-size 2`, `--grad-accum 4`, `--curriculum fullpage`, `--fullpage-max-dim 1280`, `--max-seq-len 1536`, `--bf16`, `--grad-checkpoint`
- `--stroke-hog-weight 0.15`, `--gpu-ema-decay 0.997`, `--save-every-minutes 5.0`, `--eval-every 300`, `--eval-max-len 512`, `--seed 42`

### Sweep Matrix (6 Arms Sharing Control `reg4_jit008`)

| Arm | Sub-Sweep | `n_register_tokens` | `rope_jitter` | `irope_nope_every` |
|---|---|---:|---:|---:|
| `reg0_jit008` | Registers | `0` | `0.08` | `4` |
| `reg4_jit008` | Control (shared) | `4` | `0.08` | `4` |
| `reg8_jit008` | Registers | `8` | `0.08` | `4` |
| `reg16_jit008` | Registers | `16` | `0.08` | `4` |
| `reg4_jit000` | RoPE Jitter | `4` | `0.00` | `4` |
| `reg4_jit016` | RoPE Jitter | `4` | `0.16` | `4` |

---

## 5. 5-Minute Checkpointing & Resume Plan

- Every arm runs with `--save-every-minutes 5.0`, saving atomic rolling `uni_omr_v2_latest.pt` + `uni_omr_v2_ema_latest.pt` under `/content/out/<exp_id>/<arm>/`.
- Completed arms write `<arm>/arm_done.json` and trigger an immediate HF Hub upload of `evals/<exp_id>.json`. Re-running `run_colab.py` automatically skips completed arms.

---

## 6. Evaluation Gate & Decision Tree

- **Measured Metrics per Arm:**
  - Held-out `proxy_val.jsonl` (`64` pages): `token_f1`, `ser`, `sys_count_exact` (raw & EMA), plus `final_25step_loss`, `steps_per_sec`, `peak_vram_mb`.
- **Promote:** Lock the winning `(n_register_tokens, rope_jitter)` pair into `UniOMRv2Config.small_45m()` and `edge85m()` defaults (and expose CLI flags in `train_uni_v2.py`) if it beats `reg0_jit008` or `reg4_jit008` by `>= +0.4 pp` Token F1 (or `>= +2.0 pp` `sys_count_exact`) with `< 3%` speed cost.
- **Iterate:** If `reg8` or `reg16` wins strongly AND `jit016` beats `jit008`, run a single confirmation arm combining the winning `n_register_tokens` + `rope_jitter` with `irope_nope_every in {0, 2, 4}`.
- **Discard:** If `n_register_tokens=0` and `rope_jitter=0.0` match or beat all treatments within `0.2 pp`, simplify `UniOMRv2Config` defaults to `0` / `0.0`.
