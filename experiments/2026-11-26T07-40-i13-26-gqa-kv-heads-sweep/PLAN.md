# Plan — `2026-11-26T07-40-i13-26-gqa-kv-heads-sweep`

## 1. Platform Chosen & Rationale

- **Platform:** Google Colab `L4` (24 GB VRAM, `cc=8.9`, native BF16).
- **Rationale:** 4 controlled `n_kv_heads` arms (`400` steps each at `1280px`) run cleanly in ~28–34 min (~2.6 CU) on a single L4 with identical seed, data order, and optimizer state, directly measuring the Token F1 vs. KV-cache / WebGPU-size Pareto frontier.

---

## 2. Identifiers for Follow-Up Queries

- **Experiment ID:** `2026-11-26T07-40-i13-26-gqa-kv-heads-sweep`
- **Colab Session Name (recommended):** `i13-26-gqa-kv`
- **Colab Commands:**
  - Status: `.\.venv\Scripts\colab.exe sessions`
  - Usage: `.\.venv\Scripts\colab.exe usage`
  - Stop (MANDATORY on finish): `.\.venv\Scripts\colab.exe stop -s i13-26-gqa-kv`
- **Hugging Face Private Dataset Hub (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-26T07-40-i13-26-gqa-kv-heads-sweep.json`
  - Run JSON: `runs/2026-11-26T07-40-i13-26-gqa-kv-heads-sweep.json`

---

## 3. Required Training & Eval Data

- **Dataset Tar:** `datasets/iter10_5_proxy_2k.tar` on `gregoreesmaa/sheet-music-converter-omr` (100% permissive license; zero `CC-BY-NC`, zero `suite31_hw` eval overlap).
  - `proxy_train.jsonl`: 2,000 full-page training samples.
  - `proxy_val.jsonl`: 64 held-out full-page validation samples.
- **Model Init:** Fresh `UniOMRv2Config.small_45m()` (`seed=42`) with `n_kv_heads` injected cleanly per arm via `train_uni_v2.build_v2_config` override (zero edits to `uni_omr_v2.py` or `train_uni_v2.py`).

---

## 4. Exact Launch Command & Hyperparameters

```powershell
# 1. Provision L4 session & upload code bundle + HF token (zero secrets in git)
.\.venv\Scripts\colab.exe new -s i13-26-gqa-kv --gpu L4
.\.venv\Scripts\colab.exe upload -s i13-26-gqa-kv /path/to/mvd_bundle.tar.gz /content/bundle.tar.gz
.\.venv\Scripts\colab.exe upload -s i13-26-gqa-kv /path/to/local_hf_token /tmp/.hf_token

# 2. Execute standalone experiment runner
.\.venv\Scripts\colab.exe exec -s i13-26-gqa-kv -f experiments/2026-11-26T07-40-i13-26-gqa-kv-heads-sweep/run_colab.py

# 3. Tear down immediately after completion (zero idle CU burn)
.\.venv\Scripts\colab.exe stop -s i13-26-gqa-kv
```

### Matched Hyperparameters Across All 4 Arms (`train_uni_v2.py`)

- `--preset small_45m` (`d_model=384, n_heads=8, n_dec_layers=8`), `max_mem_tokens=3072`, `keep_ratio=0.40`, `dynamic_routing=True`
- `--optimizer muon`, `--lr 3e-4`, `--weight-decay 0.01`, `--lr-schedule wsd`, `--warmup-steps 60`, `--wsd-decay-frac 0.25`, `--min-lr-ratio 0.05`
- `--steps 400`, `--batch-size 2`, `--grad-accum 4`, `--curriculum fullpage`, `--fullpage-max-dim 1280`, `--max-seq-len 1536`, `--bf16`, `--grad-checkpoint`
- `--stroke-hog-weight 0.15`, `--gpu-ema-decay 0.997`, `--save-every-minutes 5.0`, `--eval-every 400`, `--eval-max-len 512`, `--seed 42`

### Sweep Matrix (4 Arms)

| Arm | `n_kv_heads` | GQA Ratio | `kv_dim` | Analytic FP16 KV Cache (`L_self=2048, L_cross=3072`) |
|---|---:|---:|---:|---:|
| `kv1_mqa` | `1` | `8:1` | `48` | `7.5 MB` |
| `kv2_gqa4x` | `2` | `4:1` (control) | `96` | `15.0 MB` |
| `kv4_gqa2x` | `4` | `2:1` | `192` | `30.0 MB` |
| `kv8_mha` | `8` | `1:1` (MHA) | `384` | `60.0 MB` |

---

## 5. 5-Minute Checkpointing & Resume Plan

- Every arm runs with `--save-every-minutes 5.0`, writing atomic rolling `uni_omr_v2_latest.pt` + `uni_omr_v2_ema_latest.pt` under `/content/i13_26/out/<exp_id>/<arm>/`.
- Completed arms write `<arm>/arm_done.json` and immediately upload updated `evals/<exp_id>.json` to HF. Re-running `run_colab.py` skips completed arms automatically.

---

## 6. Evaluation Gate & Decision Tree

- **Measured Metrics per Arm:**
  - Held-out `proxy_val.jsonl` (`64` pages): `token_f1`, `ser`, `sys_count_exact`, `tokens_per_sec` (raw & EMA), plus `final_25step_loss`, `steps_per_sec`, `peak_vram_mb`, `kv_cache_fp16_mb`, `hybrid_fp16_int4_with_scales_mb`.
- **Promote:** Update `UniOMRv2Config` default `n_kv_heads` to `4` (or `8`) if it beats `n_kv_heads=2` by `>= +0.5 pp` EMA Token F1 while keeping Hybrid INT4 model size `< 55 MB` and decode `tokens_per_sec` within `10%`.
- **Iterate:** If `kv8_mha` beats `kv4_gqa2x` by `>= +0.5 pp`, test asymmetric GQA (`n_kv_heads_self=2`, `n_kv_heads_cross=8`) to keep self-attention KV cache tiny while unbottlenecking visual cross-attention.
- **Discard:** If `kv2_gqa4x` matches `kv4_gqa2x` and `kv8_mha` within `0.3 pp` Token F1, retain `n_kv_heads=2` for maximum WebGPU KV-cache efficiency.
