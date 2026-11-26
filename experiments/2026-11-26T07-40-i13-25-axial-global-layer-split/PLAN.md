# Plan — `2026-11-26T07-40-i13-25-axial-global-layer-split`

## 1. Platform Chosen & Rationale

- **Platform:** Google Colab `L4` (24 GB VRAM, `cc=8.9`, native BF16).
- **Rationale:** 4 iso-depth proxy arms (`400` steps each at `1280px` with axial-only gradient checkpointing) fit cleanly in ~30–36 min (~2.8 CU) on a single L4 with identical seed, data order, and optimizer state.

---

## 2. Identifiers for Follow-Up Queries

- **Experiment ID:** `2026-11-26T07-40-i13-25-axial-global-layer-split`
- **Colab Session Name (recommended):** `i13-25-ax-gl`
- **Colab Commands:**
  - Status: `.\.venv\Scripts\colab.exe sessions`
  - Usage: `.\.venv\Scripts\colab.exe usage`
  - Stop (MANDATORY on finish): `.\.venv\Scripts\colab.exe stop -s i13-25-ax-gl`
- **Hugging Face Private Dataset Hub (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-26T07-40-i13-25-axial-global-layer-split.json`
  - Run JSON: `runs/2026-11-26T07-40-i13-25-axial-global-layer-split.json`

---

## 3. Required Training & Eval Data

- **Dataset Tar:** `datasets/iter10_5_proxy_2k.tar` on `gregoreesmaa/sheet-music-converter-omr` (100% permissive license; zero `CC-BY-NC`, zero `suite31_hw` eval overlap).
  - `proxy_train.jsonl`: 2,000 full-page training samples.
  - `proxy_val.jsonl`: 64 held-out full-page validation samples.
- **Model Init:** Fresh `UniOMRv2Config.small_45m()` (`seed=42`) with `n_axial_layers` and `n_global_layers` injected cleanly per arm via `train_uni_v2.build_v2_config` override (zero edits to `uni_omr_v2.py` or `train_uni_v2.py`).

---

## 4. Exact Launch Command & Hyperparameters

```powershell
# 1. Provision L4 session & upload code bundle + HF token (zero secrets in git)
.\.venv\Scripts\colab.exe new -s i13-25-ax-gl --gpu L4
.\.venv\Scripts\colab.exe upload -s i13-25-ax-gl /path/to/mvd_bundle.tar.gz /content/bundle.tar.gz
.\.venv\Scripts\colab.exe upload -s i13-25-ax-gl /path/to/local_hf_token /tmp/.hf_token

# 2. Execute standalone experiment runner
.\.venv\Scripts\colab.exe exec -s i13-25-ax-gl -f experiments/2026-11-26T07-40-i13-25-axial-global-layer-split/run_colab.py

# 3. Tear down immediately after completion (zero idle CU burn)
.\.venv\Scripts\colab.exe stop -s i13-25-ax-gl
```

### Matched Hyperparameters Across All 4 Arms (`train_uni_v2.py`)

- `--preset small_45m`, `max_mem_tokens=3072`, `keep_ratio=0.40`, `dynamic_routing=True`
- `--optimizer muon`, `--lr 3e-4`, `--weight-decay 0.01`, `--lr-schedule wsd`, `--warmup-steps 60`, `--wsd-decay-frac 0.25`, `--min-lr-ratio 0.05`
- `--steps 400`, `--batch-size 2`, `--grad-accum 4`, `--curriculum fullpage`, `--fullpage-max-dim 1280`, `--max-seq-len 1536`, `--bf16`, `--grad-checkpoint`
- `--stroke-hog-weight 0.15`, `--gpu-ema-decay 0.997`, `--save-every-minutes 5.0`, `--eval-every 400`, `--eval-max-len 512`, `--seed 42`

### Sweep Matrix (4 Iso-Depth Arms, `n_axial + n_global = 6`)

| Arm | `n_axial_layers` | `n_global_layers` | Role |
|---|---:|---:|---|
| `ax1_gl5` | `1` | `5` | Global-heavy (`/16`) |
| `ax2_gl4` | `2` | `4` | Current default control |
| `ax3_gl3` | `3` | `3` | Balanced `/8` axial & `/16` global |
| `ax4_gl2` | `4` | `2` | High-res `/8` axial-heavy |

---

## 5. 5-Minute Checkpointing & Resume Plan

- Every arm runs with `--save-every-minutes 5.0`, writing atomic rolling `uni_omr_v2_latest.pt` + `uni_omr_v2_ema_latest.pt` under `/content/i13_25/out/<exp_id>/<arm>/`.
- Completed arms write `<arm>/arm_done.json` and immediately upload updated `evals/<exp_id>.json` to HF. Re-running `run_colab.py` skips completed arms automatically.

---

## 6. Evaluation Gate & Decision Tree

- **Measured Metrics per Arm:**
  - Held-out `proxy_val.jsonl` (`64` pages): `token_f1`, `ser`, `sys_count_exact` (raw & EMA), plus `final_25step_loss`, `steps_per_sec`, `peak_vram_mb`, `total_params`, `hybrid_fp16_int4_with_scales_mb`.
- **Promote:** Update `UniOMRv2Config.small_45m()` and `edge85m()` defaults to the winning `(n_axial_layers, n_global_layers)` split if it beats `ax2_gl4` by `>= +0.5 pp` EMA Token F1 (or `>= +2.0 pp` `sys_count_exact`) with `< 12%` step-time overhead and `< 55 MB` Hybrid INT4 size.
- **Iterate:** If `ax4_gl2` wins monotonically at the boundary, test `(n_axial=4, n_global=3)` and `axial_ffn_mult in {2.0, 3.0}`.
- **Discard:** If `ax2_gl4` ties or beats all treatments within `0.3 pp` Token F1, retain `(2, 4)` as optimal.
