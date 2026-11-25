# PLAN — `2026-11-26T01-38-ov-13-muon-lr-sweep`

Zero-prior-context operational handoff for **Overnight Exp 13: Muon+AdamW Peak LR Sweep** on `UniOMR v2 small_45m`.

---

## 1. Platform & Budget

- **Provider:** Google Colab (`L4`, 24 GB VRAM, `cc=8.9`, BF16 native, `torch 2.14.1+cu130` pre-installed)
- **Estimated Wall-Clock:** `~30 min` (`3 arms × 800 steps` = `2,400` steps + 6 `proxy_val` evals)
- **Estimated Compute Units:** `~1.5–2.4 CU` (auto-teardown on exit)
- **Baseline Commit SHA:** `164852f`

---

## 2. Identifiers & Artifact Paths

- **Experiment ID:** `2026-11-26T01-38-ov-13-muon-lr-sweep`
- **Colab Session Name (if persistent):** `ov-13-muon-lr-sweep`
- **Hugging Face Repo:** `gregoreesmaa/sheet-music-converter-omr` (`repo_type="dataset"`)
  - Input dataset: `datasets/iter10_5_proxy_2k.tar` (`2,000` train / `64` val full pages, permissive)
  - Eval JSON output: `evals/2026-11-26T01-38-ov-13-muon-lr-sweep.json`
  - Run ledger output: `runs/2026-11-26T01-38-ov-13-muon-lr-sweep.json`
  - Winning EMA checkpoint: `checkpoints/2026-11-26T01-38-ov-13-muon-lr-sweep/best_ema_latest.pt`

---

## 3. Exact Parameter Table

| Hyperparameter | Value | Notes |
|---|---|---|
| `--preset` | `small_45m` | `~43M` params (`UniOMRv2Net`) |
| `--optimizer` | `muon` | Native `torch.optim.Muon` (2D Transformer) + fused `AdamW` (rest) |
| `--lr` (swept) | `1.5e-4`, `3.0e-4`, `5.0e-4` | Single isolated variable across 3 arms |
| `--weight-decay` | `0.01` | Held constant |
| `--steps` | `800` | Per arm (`2,400` total) |
| `--batch-size` / `--grad-accum` | `2` / `4` | Effective batch size = `8` full pages |
| `--fullpage-max-dim` / `--max-seq-len` | `1280` / `2048` | Multi-scale enabled |
| `--lr-schedule` | `wsd` | `--warmup-steps 80 --wsd-decay-frac 0.25 --min-lr-ratio 0.05` |
| `--gpu-ema-decay` / `--gpu-ema-every` | `0.997` / `5` | Device-resident EMA |
| `--save-every-minutes` | `5.0` | Atomic `.tmp -> os.replace` for raw + EMA |
| `--augment` / `--aug-prob` | `True` / `0.50` | Stratified full-page sampling |
| `--bf16 --grad-checkpoint --channels-last` | Enabled | CUTLASS/Flash SDPA (`cudnn_sdp=False`) |
| `--seed` | `42` | Identical across all arms |
| Proxy Val Eval | `64` pages (`proxy_val.jsonl`) | Teacher-forced CE/MTP/HOG + greedy SER/Token F1 (`eval_max_len=512`) |

---

## 4. Copy-Paste Launch Commands (Windows PowerShell)

### Option A — Named Colab L4 Session (Recommended for Code Bundle + `/tmp/.hf_token` + Auto-Teardown)

```powershell
# 1. Create lightweight code bundle (tmp-train/mvd only, zero secrets)
tar -czf D:\tmp\ov13_bundle.tar.gz tmp-train/mvd

# 2. Provision Colab L4 session
.\.venv\Scripts\colab.exe new -s ov-13-muon-lr-sweep --gpu L4

# 3. Upload code bundle and temporary HF token (deleted in finally by run_colab.py)
.\.venv\Scripts\colab.exe upload -s ov-13-muon-lr-sweep D:\tmp\ov13_bundle.tar.gz /content/bundle.tar.gz
$env:HF_TOKEN | Out-File -FilePath D:\tmp\.hf_token_tmp -Encoding ascii -NoNewline
.\.venv\Scripts\colab.exe upload -s ov-13-muon-lr-sweep D:\tmp\.hf_token_tmp /tmp/.hf_token
Remove-Item D:\tmp\.hf_token_tmp -Force

# 4. Run experiment and ALWAYS stop session on exit (zero idle CU burn)
try {
    .\.venv\Scripts\colab.exe exec -s ov-13-muon-lr-sweep -f experiments/2026-11-26T01-38-ov-13-muon-lr-sweep/run_colab.py
} finally {
    .\.venv\Scripts\colab.exe stop -s ov-13-muon-lr-sweep
}
```

### Option B — Ephemeral One-Shot `colab run` (if bundle is already hosted/synced)

```powershell
.\.venv\Scripts\colab.exe run --gpu L4 experiments/2026-11-26T01-38-ov-13-muon-lr-sweep/run_colab.py
```

---

## 5. Health Check-In (Minute 3–5) & Resume

- Verify `GPU=NVIDIA L4 (cc=8.9)`, `mode=muon+adamw`, finite loss decreasing on arm `lr_1p5e-4`, and `[autosave @ step ...]` at minute 5.
- **Resume:** Each arm writes `/content/vol/ckpts/2026-11-26T01-38-ov-13-muon-lr-sweep/<arm>/arm_result.json` on completion and skips completed arms on restart.

---

## 6. HF Result Retrieval Commands

```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe -c "
from huggingface_hub import hf_hub_download
for p in [
    'evals/2026-11-26T01-38-ov-13-muon-lr-sweep.json',
    'runs/2026-11-26T01-38-ov-13-muon-lr-sweep.json',
]:
    out = hf_hub_download('gregoreesmaa/sheet-music-converter-omr', p, repo_type='dataset', local_dir='D:/tmp/ov_results')
    print('Downloaded:', out)
"
```

---

## 7. Decision Tree

1. **Promote winning LR** into Wave 5 Iter-11 `A100/H100` launch config if it beats `3.0e-4` by `<= -0.02` EMA val CE and `>= +0.01` EMA Token F1 with `p95 grad_norm < 5.0`.
2. **Keep `3.0e-4`** if within noise or if `5.0e-4` shows instability.
3. **Write `OUTCOME.md`** immediately after retrieving the HF JSON, scrutinizing per-100-step loss/grad curves and peak VRAM.
