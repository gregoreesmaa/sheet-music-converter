# Plan — `2026-11-26T01-38-ov-16-resolution-multiscale-sweep`

## 1. Platform Chosen & Rationale

- **Platform:** Google Colab `L4` (24 GB VRAM, `cc=8.9`, native BF16, `torch 2.14.1+cu130` pre-installed).
- **Rationale:** `1536px` full-page batches need >16 GB headroom to avoid WDDM/OOM spikes while running 3 controlled arms back-to-back in ~35–40 min (~3.2 compute units at ~4.8 CU/hr).
- **Budget:** 3 arms × `450` optimizer steps (`batch_size=2`, `grad_accum=4`, effective batch `8` full pages/step) ≈ 35–40 min total wall-clock.

---

## 2. Identifiers for Follow-Up Queries

- **Experiment ID:** `2026-11-26T01-38-ov-16-resolution-multiscale-sweep`
- **Colab Session Name (recommended):** `ov-16-res-ms`
- **Colab Commands:**
  - Status: `.\.venv\Scripts\colab.exe sessions`
  - Usage: `.\.venv\Scripts\colab.exe usage`
  - Stop (MANDATORY on finish): `.\.venv\Scripts\colab.exe stop -s ov-16-res-ms`
- **Hugging Face Private Dataset Hub (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-26T01-38-ov-16-resolution-multiscale-sweep.json`
  - Run JSON: `runs/2026-11-26T01-38-ov-16-resolution-multiscale-sweep.json`

---

## 3. Required Training & Eval Data

- **Dataset Tar:** `datasets/iter10_5_proxy_2k.tar` on `gregoreesmaa/sheet-music-converter-omr` (permissive license only; zero `CC-BY-NC`, zero `suite31_hw` eval overlap).
  - `proxy_train.jsonl`: 2,000 full-page music score training samples (printed + HW mix).
  - `proxy_val.jsonl`: 64 held-out full-page validation samples evaluated at each arm's native `fullpage_max_dim` AND at a common `1280px` reference resolution for cross-scale check.
- **Model Init:** Fresh `UniOMRv2Config.small_45m()` (`seed=42`, identical initialization across all 3 arms).

---

## 4. Exact Launch Command & Hyperparameters

```powershell
# 1. Provision L4 session & upload code bundle + HF token (zero secrets in git)
.\.venv\Scripts\colab.exe new -s ov-16-res-ms --gpu L4
.\.venv\Scripts\colab.exe upload -s ov-16-res-ms /path/to/mvd_bundle.tar.gz /content/bundle.tar.gz
.\.venv\Scripts\colab.exe upload -s ov-16-res-ms /path/to/local_hf_token /tmp/.hf_token

# 2. Execute standalone experiment runner
.\.venv\Scripts\colab.exe exec -s ov-16-res-ms -f experiments/2026-11-26T01-38-ov-16-resolution-multiscale-sweep/run_colab.py

# 3. Tear down immediately after completion (zero idle CU burn)
.\.venv\Scripts\colab.exe stop -s ov-16-res-ms
```

### Matched Hyperparameters Across All 3 Arms (`train_uni_v2.py`)

- `--preset small_45m` (`d_model=384`, `n_heads=8`, `n_kv_heads=2`, `n_axial_layers=2`, `n_global_layers=4`, `n_dec_layers=8`, `n_register_tokens=4`, `rope_jitter=0.08`, `irope_nope_every=4`)
- `--optimizer muon`, `--lr 3e-4`, `--weight-decay 0.01`, `--lr-schedule wsd`, `--warmup-steps 60`, `--wsd-decay-frac 0.25`, `--min-lr-ratio 0.05`
- `--steps 450`, `--batch-size 2`, `--grad-accum 4`, `--curriculum fullpage`, `--max-seq-len 1536`, `--bf16`, `--grad-checkpoint`
- `--keep-ratio 0.40`, `--dynamic-routing`, `--max-mem-tokens 10240`
- `--stroke-hog-weight 0.15`, `--mtp-loss-weight 0.30`, `--saliency-loss-weight 0.10`, `--sys-y-loss-weight 0.25`, `--sys-count-loss-weight 0.15`, `--staff-seg-loss-weight 0.20`
- `--gpu-ema-decay 0.997`, `--save-every-minutes 5.0`, `--eval-every 450`, `--eval-max-len 512`, `--seed 42`

### Swept Variable (1 Variable Only: Resolution / Multi-Scale)

| Arm | `--fullpage-max-dim` | `multi_scale` | `prefetch_threads` | `num_workers` |
|---|---:|---|---:|---:|
| `1024px_fixed` | `1024` | `False` | `4` | `0` |
| `1280px_fixed` | `1280` | `False` | `4` | `0` |
| `1536px_multiscale` | `1536` | `True` (`MultiScaleBatchSampler`) | `0` | `2` |

---

## 5. 5-Minute Checkpointing & Resume Plan

- Every arm invokes `train_uni_v2.run_train_v2` with `--save-every-minutes 5.0`, triggering atomic `.tmp -> os.replace` rolling saves of `uni_omr_v2_latest.pt` and `uni_omr_v2_ema_latest.pt` under `/content/out/<exp_id>/<arm>/`.
- After each arm completes, `run_colab.py` writes incremental JSON results to `/content/out/<exp_id>/results.json` and uploads `evals/<exp_id>.json` + `runs/<exp_id>.json` to HF Private Hub so even a mid-sweep disconnect preserves completed arms.
- **Resume if preempted:** Re-run `run_colab.py` in the same session — any arm with `<arm>/arm_done.json` on disk is skipped automatically, or pass `--init-from /content/out/<exp_id>/<arm>/uni_omr_v2_latest.pt`.

---

## 6. Evaluation Gate & Decision Tree

- **Measured Metrics per Arm:**
  1. Held-out `proxy_val.jsonl` (`64` pages): `token_f1`, `ser`, `sys_count_exact`, `mean_compression_x` (raw & EMA).
  2. Router token budget: mean `n_in`, mean `n_out`, effective compression ratio `n_in / n_out`.
  3. Throughput & VRAM: `steps_per_sec`, `sec_per_step`, `peak_vram_mb`.
- **Promote:** If `1536px_multiscale` beats `1280px_fixed` by `>= +0.5 pp` `token_f1` (or `>= -0.005` SER) with `< 1.6x` wall-clock slowdown and `< 20 GB` peak VRAM, make `1536px_multiscale` the default full-page recipe for Iter-11 `UniOMR v2`.
- **Iterate:** If `1536px_multiscale` improves F1 but exceeds `1.6x` step time, follow up with a `keep_ratio=0.30` / `max_mem_tokens=4096` token-cap ablation at `1536px`.
- **Discard:** If `1024px_fixed` matches `1280px_fixed` within `0.3 pp` F1 at `>= 1.4x` speedup, keep `1280px` as default and use `1024px` for fast proxy ablations.
