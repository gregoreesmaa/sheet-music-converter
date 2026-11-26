# PLAN — `2026-11-26T07-40-i13-29-grad-clip-threshold-sweep`

Zero-prior-context operational handoff for **Iter 13 Exp 29: Global Gradient Clipping Threshold Sweep** on `UniOMR v2 small_45m`.

---

## 1. Platform & Budget
- **Provider:** Google Colab (`L4`, 24 GB VRAM, `cc=8.9`, BF16 native, native `torch.optim.Muon`)
- **Estimated Wall-Clock:** `~28 min` (`4 arms x 600 steps` = `2,400` steps + `8` `proxy_val` evals)
- **Estimated Compute Units:** `~2.3 CU` (auto-teardown on exit)
- **Baseline Commit SHA:** `9303e51`

---

## 2. Identifiers & Artifact Paths
- **Experiment ID:** `2026-11-26T07-40-i13-29-grad-clip-threshold-sweep`
- **Colab Session:** `i13-29-grad-clip`
- **HF Repo (`dataset`):** `gregoreesmaa/sheet-music-converter-omr`
  - Input dataset: `datasets/iter10_5_proxy_2k.tar`
  - Eval JSON: `evals/2026-11-26T07-40-i13-29-grad-clip-threshold-sweep.json`
  - Run ledger: `runs/2026-11-26T07-40-i13-29-grad-clip-threshold-sweep.json`

### Copy-Paste Monitoring & Control Commands (PowerShell)
```powershell
.\.venv\Scripts\colab.exe sessions
.\.venv\Scripts\colab.exe usage
.\.venv\Scripts\colab.exe stop -s i13-29-grad-clip

$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; p = hf_hub_download('gregoreesmaa/sheet-music-converter-omr', 'evals/2026-11-26T07-40-i13-29-grad-clip-threshold-sweep.json', repo_type='dataset'); print(open(p, encoding='utf-8').read())"
```

---

## 3. Exact Parameter Table (Isolated Variable: `grad_clip` max L2 norm)

| Arm | `grad_clip` (`max_norm`) | Notes |
|---|---:|---|
| `clip_0p5` | `0.5` | Stricter clip (tests if tighter clipping stabilizes early spikes) |
| `clip_1p0` | `1.0` | Current hardcoded `train_uni_v2.py:2099` default (control) |
| `clip_2p0` | `2.0` | Relaxed 2x threshold |
| `clip_5p0` | `5.0` | Relaxed 5x threshold (tests if `1.0` was choking backbone/decoder grads) |

### Matched Hyperparameters Across All 4 Arms
- `--preset small_45m`, `--optimizer muon`, `--lr 3e-4`, `--weight-decay 0.01`, `--lr-schedule wsd`, `--warmup-steps 60`, `--wsd-decay-frac 0.25`, `--min-lr-ratio 0.05`
- `--steps 600`, `--batch-size 2`, `--grad-accum 4` (eff batch = `8`), `--curriculum fullpage`, `--fullpage-max-dim 1280`, `--max-seq-len 2048`, `--multi-scale`, `--augment --aug-prob 0.50`
- `--grad-checkpoint --channels-last --bf16`, `--gpu-ema-decay 0.99 --gpu-ema-every 1`, `--save-every-minutes 5.0`, `--seed 42`
- Telemetry per 100 steps: pre-clip `grad_norm_mean`, `grad_norm_p95`, `clip_rate` (fraction of steps where pre-clip norm `> max_norm`), plus `nonfinite_steps`.

---

## 4. Decision Tree
1. **PROMOTE:** If `clip_2p0` or `clip_5p0` (or `clip_0p5`) beats `clip_1p0` by `>= +0.005` EMA `val_token_f1` OR `<= -0.015` EMA `val_ce_loss` with `0` NaN/Inf steps -> expose `--grad-clip` CLI flag in `train_uni_v2.py` with the winning default.
2. **ITERATE:** If `clip_1p0` has `clip_rate > 0.80` AND `clip_5p0` still has high `clip_rate` without instability -> test decoupled clipping (clip AdamW params separately from Muon 2D params).
3. **DISCARD:** If `clip_1p0` matches or beats all alternatives -> keep `1.0` (still worth exposing as `--grad-clip 1.0`).
