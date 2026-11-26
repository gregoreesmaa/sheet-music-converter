# Plan — `2026-11-26T07-40-i13-27-rope-coord-scale-sweep`

## 1. Platform Chosen & Rationale

- **Platform:** Google Colab `L4` (24 GB VRAM, `cc=8.9`, native BF16).
- **Rationale:** 4 zero-param-overhead `rope_coord_scale` arms (`400` steps each at `1280px`) run cleanly in ~28–34 min (~2.6 CU) on a single L4 with identical seed, data order, and optimizer state.

---

## 2. Identifiers for Follow-Up Queries

- **Experiment ID:** `2026-11-26T07-40-i13-27-rope-coord-scale-sweep`
- **Colab Session Name (recommended):** `i13-27-rope-scale`
- **Colab Commands:**
  - Status: `.\.venv\Scripts\colab.exe sessions`
  - Usage: `.\.venv\Scripts\colab.exe usage`
  - Stop (MANDATORY on finish): `.\.venv\Scripts\colab.exe stop -s i13-27-rope-scale`
- **Hugging Face Private Dataset Hub (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-26T07-40-i13-27-rope-coord-scale-sweep.json`
  - Run JSON: `runs/2026-11-26T07-40-i13-27-rope-coord-scale-sweep.json`

---

## 3. Required Training & Eval Data

- **Dataset Tar:** `datasets/iter10_5_proxy_2k.tar` on `gregoreesmaa/sheet-music-converter-omr` (100% permissive license; zero `CC-BY-NC`, zero `suite31_hw` eval overlap).
  - `proxy_train.jsonl`: 2,000 full-page training samples.
  - `proxy_val.jsonl`: 64 held-out full-page validation samples.
- **Model Init:** Fresh `UniOMRv2Config.small_45m()` (`seed=42`) with `rope_coord_scale` injected cleanly per arm via `train_uni_v2.build_v2_config` override (zero edits to `uni_omr_v2.py` or `train_uni_v2.py`).

---

## 4. Exact Launch Command & Hyperparameters

```powershell
# 1. Provision L4 session & upload code bundle + HF token (zero secrets in git)
.\.venv\Scripts\colab.exe new -s i13-27-rope-scale --gpu L4
.\.venv\Scripts\colab.exe upload -s i13-27-rope-scale /path/to/mvd_bundle.tar.gz /content/bundle.tar.gz
.\.venv\Scripts\colab.exe upload -s i13-27-rope-scale /path/to/local_hf_token /tmp/.hf_token

# 2. Execute standalone experiment runner
.\.venv\Scripts\colab.exe exec -s i13-27-rope-scale -f experiments/2026-11-26T07-40-i13-27-rope-coord-scale-sweep/run_colab.py

# 3. Tear down immediately after completion (zero idle CU burn)
.\.venv\Scripts\colab.exe stop -s i13-27-rope-scale
```

### Matched Hyperparameters Across All 4 Arms (`train_uni_v2.py`)

- `--preset small_45m` (`rope_base=10000.0`, `rope_jitter=0.08`, `n_register_tokens=4`), `max_mem_tokens=3072`, `keep_ratio=0.40`, `dynamic_routing=True`
- `--optimizer muon`, `--lr 3e-4`, `--weight-decay 0.01`, `--lr-schedule wsd`, `--warmup-steps 60`, `--wsd-decay-frac 0.25`, `--min-lr-ratio 0.05`
- `--steps 400`, `--batch-size 2`, `--grad-accum 4`, `--curriculum fullpage`, `--fullpage-max-dim 1280`, `--max-seq-len 1536`, `--bf16`, `--grad-checkpoint`
- `--stroke-hog-weight 0.15`, `--gpu-ema-decay 0.997`, `--save-every-minutes 5.0`, `--eval-every 400`, `--eval-max-len 512`, `--seed 42`

### Sweep Matrix (4 Arms, Zero Parameter/Speed Cost)

| Arm | `rope_coord_scale` | Max `/16` Phase Step (`h_p=80`) | Regime |
|---|---:|---:|---|
| `scale_32` | `32.0` | `0.41 rad/patch` | Smooth sub-Nyquist |
| `scale_64` | `64.0` | `0.81 rad/patch` | Grid-matched sub-Nyquist |
| `scale_128` | `128.0` | `1.62 rad/patch` | Current default control (`> 90°/patch`) |
| `scale_256` | `256.0` | `3.24 rad/patch` | Aliased (`> pi` per patch) |

---

## 5. 5-Minute Checkpointing & Resume Plan

- Every arm runs with `--save-every-minutes 5.0`, writing atomic rolling `uni_omr_v2_latest.pt` + `uni_omr_v2_ema_latest.pt` under `/content/i13_27/out/<exp_id>/<arm>/`.
- Completed arms write `<arm>/arm_done.json` and immediately upload updated `evals/<exp_id>.json` to HF. Re-running `run_colab.py` skips completed arms automatically.

---

## 6. Evaluation Gate & Decision Tree

- **Measured Metrics per Arm:**
  - Held-out `proxy_val.jsonl` (`64` pages): `token_f1`, `ser`, `sys_count_exact` (raw & EMA), plus `final_25step_loss`, `steps_per_sec`, `peak_vram_mb`.
- **Promote:** Update `UniOMRv2Config.rope_coord_scale` default (currently `128.0`) to the winning value if it beats `scale_128` by `>= +0.4 pp` EMA Token F1 (zero inference/memory cost!).
- **Iterate:** If `scale_32` wins at the lower boundary, test `rope_coord_scale in {16.0, 24.0, 32.0}` and decouple axial `/8` vs global `/16` coordinate scales (`128.0` on `/8`, `64.0` on `/16`).
- **Discard:** If `scale_128` ties or beats all treatments within `0.2 pp` Token F1, retain `128.0`.
