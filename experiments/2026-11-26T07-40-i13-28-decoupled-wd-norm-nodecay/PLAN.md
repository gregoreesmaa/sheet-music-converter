# PLAN — `2026-11-26T07-40-i13-28-decoupled-wd-norm-nodecay`

Zero-prior-context operational handoff for **Iter 13 Exp 28: Decoupled Muon/AdamW Weight Decay & 1D Norm/Bias No-Decay Sweep** on `UniOMR v2 small_45m`.

---

## 1. Platform & Budget
- **Provider:** Google Colab (`L4`, 24 GB VRAM, `cc=8.9`, BF16 native, native `torch.optim.Muon`)
- **Estimated Wall-Clock:** `~30 min` (`4 arms x 600 steps` = `2,400` steps + `8` `proxy_val` evals)
- **Estimated Compute Units:** `~2.4 CU` (auto-teardown on exit)
- **Baseline Commit SHA:** `9303e51`

---

## 2. Identifiers & Artifact Paths
- **Experiment ID:** `2026-11-26T07-40-i13-28-decoupled-wd-norm-nodecay`
- **Colab Session:** `i13-28-wd-nodecay`
- **HF Repo (`dataset`):** `gregoreesmaa/sheet-music-converter-omr`
  - Input dataset: `datasets/iter10_5_proxy_2k.tar`
  - Eval JSON: `evals/2026-11-26T07-40-i13-28-decoupled-wd-norm-nodecay.json`
  - Run ledger: `runs/2026-11-26T07-40-i13-28-decoupled-wd-norm-nodecay.json`

### Copy-Paste Monitoring & Control Commands (PowerShell)
```powershell
.\.venv\Scripts\colab.exe sessions
.\.venv\Scripts\colab.exe usage
.\.venv\Scripts\colab.exe stop -s i13-28-wd-nodecay

$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; p = hf_hub_download('gregoreesmaa/sheet-music-converter-omr', 'evals/2026-11-26T07-40-i13-28-decoupled-wd-norm-nodecay.json', repo_type='dataset'); print(open(p, encoding='utf-8').read())"
```

---

## 3. Exact Parameter Table (Isolated Variable: Weight Decay Partitioning)

| Arm | `muon_wd` | `adamw_decay_wd` (2D+/4D weights) | `adamw_nodecay_wd` (1D norms/biases/emb) | Notes |
|---|---:|---:|---:|---|
| `wd_uniform_0p01` | `0.01` | `0.01` | `0.01` | Current `train_uni_v2.py` default (control) |
| `wd_nodecay1d_0p01` | `0.01` | `0.01` | `0.00` | Isolates pure 1D norm/bias/emb no-decay fix |
| `wd_nodecay1d_muon0p05` | `0.05` | `0.01` | `0.00` | Decoupled Muon WD `0.05` + 1D no-decay |
| `wd_nodecay1d_muon0p10` | `0.10` | `0.04` | `0.00` | Moonlight/Kimi-K2 high-WD Muon + 1D no-decay |

### Matched Hyperparameters Across All 4 Arms
- `--preset small_45m`, `--optimizer muon`, `--lr 3e-4`, `--lr-schedule wsd`, `--warmup-steps 60`, `--wsd-decay-frac 0.25`, `--min-lr-ratio 0.05`
- `--steps 600`, `--batch-size 2`, `--grad-accum 4` (eff batch = `8`), `--curriculum fullpage`, `--fullpage-max-dim 1280`, `--max-seq-len 2048`, `--multi-scale`, `--augment --aug-prob 0.50`
- `--grad-checkpoint --channels-last --bf16`, `--gpu-ema-decay 0.99 --gpu-ema-every 1`, `--save-every-minutes 5.0`, `--seed 42`
- `proxy_val` (`64` pages, `eval_max_len=512`): both RAW and EMA evaluated with unweighted `val_ce_loss`, `val_token_f1`, `val_ser`, `val_sys_count_exact`, plus final 2D weight L2 norms.

---

## 4. Decision Tree
1. **PROMOTE:** If any decoupled/no-decay arm improves EMA `val_token_f1` by `>= +0.005` OR lowers `val_ce_loss` by `>= 0.015` vs `wd_uniform_0p01` without raising `grad_norm_p95 > 5.0` -> patch `build_optimizer_v2` in `train_uni_v2.py` to split `no_decay` (`ndim < 2` or `tok_emb`) and add `--muon-weight-decay` with the winning default.
2. **ITERATE:** If `wd_nodecay1d_muon0p10` wins monotonically at the upper boundary -> probe `muon_wd in {0.10, 0.15, 0.20}` in a quick follow-up.
3. **DISCARD:** If `wd_uniform_0p01` matches or beats all 3 variants -> keep `build_optimizer_v2` unchanged.
