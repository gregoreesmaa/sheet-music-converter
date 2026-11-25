# PLAN — `2026-11-26T01-38-ov-15-ema-decay-soup-sweep`

Zero-prior-context operational handoff for **Overnight Exp 15: GPU EMA Decay & `k=3` Checkpoint Souping Sweep** on `UniOMR v2 small_45m`.

---

## 1. Platform & Budget

- **Provider:** Google Colab (`L4`, 24 GB VRAM, `cc=8.9`, BF16 native, `torch 2.14.1+cu130` pre-installed)
- **Estimated Wall-Clock:** `~32 min` (`3 arms × 800 steps` = `2,400` steps + `10` `proxy_val` evals: raw/EMA/traj-soup per arm + cross-arm EMA soup)
- **Estimated Compute Units:** `~1.6–2.5 CU` (auto-teardown on exit)
- **Baseline Commit SHA:** `164852f`

---

## 2. Identifiers & Artifact Paths

- **Experiment ID:** `2026-11-26T01-38-ov-15-ema-decay-soup-sweep`
- **Colab Session Name (if persistent):** `ov-15-ema-decay-soup-sweep`
- **Hugging Face Repo:** `gregoreesmaa/sheet-music-converter-omr` (`repo_type="dataset"`)
  - Input dataset: `datasets/iter10_5_proxy_2k.tar` (`2,000` train / `64` val full pages, permissive)
  - Eval JSON output: `evals/2026-11-26T01-38-ov-15-ema-decay-soup-sweep.json`
  - Run ledger output: `runs/2026-11-26T01-38-ov-15-ema-decay-soup-sweep.json`
  - Winning checkpoint: `checkpoints/2026-11-26T01-38-ov-15-ema-decay-soup-sweep/best_candidate.pt`

---

## 3. Exact Parameter Table

| Hyperparameter | Value | Notes |
|---|---|---|
| `--preset` | `small_45m` | `~43M` params (`UniOMRv2Net`) |
| `--optimizer` | `muon` | Native `torch.optim.Muon` + fused `AdamW` |
| `--lr` | `3.0e-4` | WSD (`--warmup-steps 80 --wsd-decay-frac 0.25 --min-lr-ratio 0.05`) |
| `--gpu-ema-decay` (swept) | `0.995`, `0.9985`, `0.9995` | Single isolated variable across 3 arms |
| `--gpu-ema-every` | `5` | Held constant |
| `k=3` Trajectory Soup | Steps `600, 700, 800` | Captured cleanly via `GPUEmaShadow.update` hook and averaged in FP32 |
| `k=3` Cross-EMA Soup | Avg of all 3 final EMA ckpts | Zero-cost post-hoc weight-space average |
| `--steps` | `800` | Per arm (`2,400` total) |
| `--batch-size` / `--grad-accum` | `2` / `4` | Effective batch size = `8` full pages |
| `--fullpage-max-dim` / `--max-seq-len` | `1280` / `2048` | Multi-scale enabled |
| `--save-every-minutes` | `5.0` | Atomic `.tmp -> os.replace` for raw + EMA |
| `--bf16 --grad-checkpoint --channels-last` | Enabled | CUTLASS/Flash SDPA (`cudnn_sdp=False`) |
| `--seed` | `42` | Identical across all arms |
| Proxy Val Eval | `64` pages (`proxy_val.jsonl`) | Teacher-forced CE/MTP/HOG + greedy SER/Token F1 (`eval_max_len=512`) |

---

## 4. Copy-Paste Launch Commands (Windows PowerShell)

```powershell
# 1. Create lightweight code bundle (tmp-train/mvd only, zero secrets)
tar -czf D:\tmp\ov15_bundle.tar.gz tmp-train/mvd

# 2. Provision Colab L4 session
.\.venv\Scripts\colab.exe new -s ov-15-ema-decay-soup-sweep --gpu L4

# 3. Upload code bundle and temporary HF token (deleted in finally by run_colab.py)
.\.venv\Scripts\colab.exe upload -s ov-15-ema-decay-soup-sweep D:\tmp\ov15_bundle.tar.gz /content/bundle.tar.gz
$env:HF_TOKEN | Out-File -FilePath D:\tmp\.hf_token_tmp -Encoding ascii -NoNewline
.\.venv\Scripts\colab.exe upload -s ov-15-ema-decay-soup-sweep D:\tmp\.hf_token_tmp /tmp/.hf_token
Remove-Item D:\tmp\.hf_token_tmp -Force

# 4. Run experiment and ALWAYS stop session on exit (zero idle CU burn)
try {
    .\.venv\Scripts\colab.exe exec -s ov-15-ema-decay-soup-sweep -f experiments/2026-11-26T01-38-ov-15-ema-decay-soup-sweep/run_colab.py
} finally {
    .\.venv\Scripts\colab.exe stop -s ov-15-ema-decay-soup-sweep
}
```

---

## 5. Health Check-In (Minute 3–5) & Resume

- Verify `GPU=NVIDIA L4 (cc=8.9)`, `mode=muon+adamw`, `[soup-snap]` logged at steps `600, 700, 800`, and `[autosave @ step ...]` at minute 5.
- **Resume:** Completed arms are cached at `/content/vol/ckpts/2026-11-26T01-38-ov-15-ema-decay-soup-sweep/<arm>/arm_result.json`.

---

## 6. HF Result Retrieval Commands

```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe -c "
from huggingface_hub import hf_hub_download
for p in [
    'evals/2026-11-26T01-38-ov-15-ema-decay-soup-sweep.json',
    'runs/2026-11-26T01-38-ov-15-ema-decay-soup-sweep.json',
]:
    out = hf_hub_download('gregoreesmaa/sheet-music-converter-omr', p, repo_type='dataset', local_dir='D:/tmp/ov_results')
    print('Downloaded:', out)
"
```

---

## 7. Decision Tree

1. **Promote winning EMA decay** into Wave 5 Iter-11 `train_uni_v2.py` launch config.
2. **Promote `k=3` checkpoint souping** into `train_uni_v2.py` (with CPU unit test) if either trajectory soup or cross-EMA soup beats the best single EMA by `<= -0.01` `proxy_val` CE without regressing Token F1.
3. **Write `OUTCOME.md`** comparing all 10 evaluated candidates (3× raw, 3× EMA, 3× `k=3` trajectory soup, 1× `k=3` cross-EMA soup).
