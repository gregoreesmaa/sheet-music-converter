# PLAN — `2026-11-26T07-40-i13-30-tok-dropout-exposure-bias-sweep`

Zero-prior-context operational handoff for **Iter 13 Exp 30: Decoder Token Dropout (`--tok-dropout`) Zero-Cost Exposure-Bias Sweep** on `UniOMR v2 small_45m`.

---

## 1. Platform & Budget
- **Provider:** Google Colab (`L4`, 24 GB VRAM, `cc=8.9`, BF16 native, native `torch.optim.Muon`)
- **Estimated Wall-Clock:** `~30 min` (`4 arms x 600 steps` = `2,400` steps + `8` long-horizon `eval_max_len=1024` `proxy_val` evals)
- **Estimated Compute Units:** `~2.4 CU` (auto-teardown on exit)
- **Baseline Commit SHA:** `9303e51`

---

## 2. Identifiers & Artifact Paths
- **Experiment ID:** `2026-11-26T07-40-i13-30-tok-dropout-exposure-bias-sweep`
- **Colab Session:** `i13-30-tok-dropout`
- **HF Repo (`dataset`):** `gregoreesmaa/sheet-music-converter-omr`
  - Input dataset: `datasets/iter10_5_proxy_2k.tar`
  - Eval JSON: `evals/2026-11-26T07-40-i13-30-tok-dropout-exposure-bias-sweep.json`
  - Run ledger: `runs/2026-11-26T07-40-i13-30-tok-dropout-exposure-bias-sweep.json`

### Copy-Paste Monitoring & Control Commands (PowerShell)
```powershell
.\.venv\Scripts\colab.exe sessions
.\.venv\Scripts\colab.exe usage
.\.venv\Scripts\colab.exe stop -s i13-30-tok-dropout

$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; p = hf_hub_download('gregoreesmaa/sheet-music-converter-omr', 'evals/2026-11-26T07-40-i13-30-tok-dropout-exposure-bias-sweep.json', repo_type='dataset'); print(open(p, encoding='utf-8').read())"
```

---

## 3. Exact Parameter Table (Isolated Variable: `--tok-dropout`)

| Arm | `--tok-dropout` | Notes |
|---|---:|---|
| `tokdrop_0p00` | `0.00` | Pure teacher forcing (no input token corruption) |
| `tokdrop_0p05` | `0.05` | Current `train_uni_v2.py:2250` default (control) |
| `tokdrop_0p15` | `0.15` | Moderate zero-cost exposure-bias regularization |
| `tokdrop_0p25` | `0.25` | Strong zero-cost exposure-bias regularization |

### Matched Hyperparameters Across All 4 Arms
- `--preset small_45m`, `--optimizer muon`, `--lr 3e-4`, `--weight-decay 0.01`, `--lr-schedule wsd`, `--warmup-steps 60`, `--wsd-decay-frac 0.25`, `--min-lr-ratio 0.05`, `--cass-prob 0.0`
- `--steps 600`, `--batch-size 2`, `--grad-accum 4` (eff batch = `8`), `--curriculum fullpage`, `--fullpage-max-dim 1280`, `--max-seq-len 2048`, `--multi-scale`, `--augment --aug-prob 0.50`
- `--grad-checkpoint --channels-last --bf16`, `--gpu-ema-decay 0.99 --gpu-ema-every 1`, `--save-every-minutes 5.0`, `--seed 42`
- **Long-page exposure-bias evaluation (`proxy_val`, 64 pages, `eval_max_len=1024`):** overall `val_token_f1`, `val_ser`, `val_ce_loss`, `val_sys_count_exact`, PLUS length-bucketed Token F1 (`short <= 256`, `medium 257..512`, `long > 512`), `mean_len_ratio` (`pred_len / ref_len`), and `early_eos_rate` (`pred_len < 0.75 * ref_len`).

---

## 4. Decision Tree
1. **PROMOTE:** If a non-default `--tok-dropout` beats `0.05` by `>= +0.005` overall EMA `val_token_f1` OR `>= +0.015` on `long > 512` Token F1 without hurting `short <= 256` -> update default `--tok-dropout` in `train_uni_v2.py`.
2. **ITERATE:** If `0.25` wins monotonically on `long > 512` F1 without underfitting -> test `0.30` and compare against CASS (`ov-25`) on long-page F1 per GPU-hour.
3. **DISCARD:** If `0.05` remains optimal -> lock `--tok-dropout 0.05` as verified default.
