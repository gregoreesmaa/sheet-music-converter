# PLAN — `2026-11-26T01-38-ov-14-wsd-cooldown-shape`

Zero-prior-context operational handoff for **Overnight Exp 14: WSD Cooldown Fraction & Decay Curve Sweep** on `UniOMR v2 small_45m`.

---

## 1. Platform & Budget

- **Provider:** Google Colab (`L4`, 24 GB VRAM, `cc=8.9`, BF16 native, `torch 2.14.1+cu130` pre-installed)
- **Estimated Wall-Clock:** `~35 min` (`6 arms × 500 steps` = `3,000` steps + 12 `proxy_val` evals)
- **Estimated Compute Units:** `~1.8–2.8 CU` (auto-teardown on exit)
- **Baseline Commit SHA:** `164852f`

---

## 2. Identifiers & Artifact Paths

- **Experiment ID:** `2026-11-26T01-38-ov-14-wsd-cooldown-shape`
- **Colab Session Name (if persistent):** `ov-14-wsd-cooldown-shape`
- **Hugging Face Repo:** `gregoreesmaa/sheet-music-converter-omr` (`repo_type="dataset"`)
  - Input dataset: `datasets/iter10_5_proxy_2k.tar` (`2,000` train / `64` val full pages, permissive)
  - Eval JSON output: `evals/2026-11-26T01-38-ov-14-wsd-cooldown-shape.json`
  - Run ledger output: `runs/2026-11-26T01-38-ov-14-wsd-cooldown-shape.json`
  - Winning EMA checkpoint: `checkpoints/2026-11-26T01-38-ov-14-wsd-cooldown-shape/best_ema_latest.pt`

---

## 3. Exact Parameter Table

| Hyperparameter | Value | Notes |
|---|---|---|
| `--preset` | `small_45m` | `~43M` params (`UniOMRv2Net`) |
| `--optimizer` | `muon` | Native `torch.optim.Muon` + fused `AdamW` |
| `--lr` | `3.0e-4` | Held constant across all 6 arms |
| `--wsd-decay-frac` (swept) | `0.15`, `0.30` | Factorial axis 1 |
| Cooldown curve (swept) | `cosine`, `linear`, `1_minus_sqrt` | Factorial axis 2 (scoped monkey-patch of `tv2._lr_for_step`) |
| `--steps` / `--warmup-steps` | `500` / `50` | `75` decay steps (`0.15`) vs `150` decay steps (`0.30`) |
| `--min-lr-ratio` | `0.05` | Final LR = `1.5e-5` at step `500` |
| `--batch-size` / `--grad-accum` | `2` / `4` | Effective batch size = `8` full pages |
| `--fullpage-max-dim` / `--max-seq-len` | `1280` / `2048` | Multi-scale enabled |
| `--gpu-ema-decay` / `--gpu-ema-every` | `0.997` / `5` | Device-resident EMA |
| `--save-every-minutes` | `5.0` | Atomic `.tmp -> os.replace` for raw + EMA |
| `--bf16 --grad-checkpoint --channels-last` | Enabled | CUTLASS/Flash SDPA (`cudnn_sdp=False`) |
| `--seed` | `42` | Identical across all 6 arms |
| Proxy Val Eval | `64` pages (`proxy_val.jsonl`) | Teacher-forced CE/MTP/HOG + greedy SER/Token F1 (`eval_max_len=512`) |

---

## 4. Copy-Paste Launch Commands (Windows PowerShell)

```powershell
# 1. Create lightweight code bundle (tmp-train/mvd only, zero secrets)
tar -czf D:\tmp\ov14_bundle.tar.gz tmp-train/mvd

# 2. Provision Colab L4 session
.\.venv\Scripts\colab.exe new -s ov-14-wsd-cooldown-shape --gpu L4

# 3. Upload code bundle and temporary HF token (deleted in finally by run_colab.py)
.\.venv\Scripts\colab.exe upload -s ov-14-wsd-cooldown-shape D:\tmp\ov14_bundle.tar.gz /content/bundle.tar.gz
$env:HF_TOKEN | Out-File -FilePath D:\tmp\.hf_token_tmp -Encoding ascii -NoNewline
.\.venv\Scripts\colab.exe upload -s ov-14-wsd-cooldown-shape D:\tmp\.hf_token_tmp /tmp/.hf_token
Remove-Item D:\tmp\.hf_token_tmp -Force

# 4. Run experiment and ALWAYS stop session on exit (zero idle CU burn)
try {
    .\.venv\Scripts\colab.exe exec -s ov-14-wsd-cooldown-shape -f experiments/2026-11-26T01-38-ov-14-wsd-cooldown-shape/run_colab.py
} finally {
    .\.venv\Scripts\colab.exe stop -s ov-14-wsd-cooldown-shape
}
```

---

## 5. Health Check-In (Minute 3–5) & Resume

- Verify `GPU=NVIDIA L4 (cc=8.9)`, `mode=muon+adamw`, exact LR trajectory at step 450/500 matching each curve, and `[autosave @ step ...]` at minute 5.
- **Resume:** Completed arms are cached at `/content/vol/ckpts/2026-11-26T01-38-ov-14-wsd-cooldown-shape/<arm>/arm_result.json`.

---

## 6. HF Result Retrieval Commands

```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe -c "
from huggingface_hub import hf_hub_download
for p in [
    'evals/2026-11-26T01-38-ov-14-wsd-cooldown-shape.json',
    'runs/2026-11-26T01-38-ov-14-wsd-cooldown-shape.json',
]:
    out = hf_hub_download('gregoreesmaa/sheet-music-converter-omr', p, repo_type='dataset', local_dir='D:/tmp/ov_results')
    print('Downloaded:', out)
"
```

---

## 7. Decision Tree

1. **Promote winning curve & fraction** by adding `--wsd-decay-curve {cosine,linear,sqrt}` to `tmp-train/mvd/uni_omr.py` + `train_uni_v2.py` (with unit tests) if non-cosine wins by `<= -0.02` EMA val CE and `>= +0.005` Token F1.
2. **Keep `cosine`** if within noise.
3. **Write `OUTCOME.md`** with full `2 × 3` main-effect and interaction table after HF retrieval.
