# Experiment Plan — `2026-11-26T00-54-ablation-vir-vico-router`

> **Zero-Prior-Context Handoff:** Everything needed to monitor, inspect, cancel, resume, or evaluate this experiment is self-contained in this folder (`README.md`, `PLAN.md`, `run_modal.py`).

## 1. Platform Chosen & Rationale
- **Provider:** Modal (`profile: gregoreesmaa1`, `$30/mo` free Starter tier)
- **Hardware:** `2× L4` (`24 GB` VRAM each, Ada `cc=8.9`, native BF16) running both arms in parallel, coordinated by a lightweight remote CPU orchestrator.
- **Estimated Wall-Clock & Cost:** ~20–30 minutes wall-clock (`~0.8–1.0` GPU-hr total across 2× L4 @ `~$0.80/hr` = **`~$0.65–$0.80`** total credit).
- **Rationale:** Native BF16 (`cc=8.9`), fast `Muon + AdamW` fused CUDA support, and parallel 2-arm fan-out without touching the Local RTX 4080 SUPER (which is busy with Iter 10.5).

## 2. Identifiers for Follow-Up Queries
- **Modal Profile:** `gregoreesmaa1`
- **Modal App Name:** `exp3-ablation-vir-vico-router`
- **Modal App ID:** `ap-s2HoXUXOCRrpAI3CMHFQJF`
- **Modal Orchestrator Call ID:** `fc-01M8AF0D8N8XRAJGEXJAP8NJ7B`
- **Modal Dashboard URL:** `https://modal.com/apps/gregoreesmaa1/main/ap-s2HoXUXOCRrpAI3CMHFQJF`
- **Modal Volume:** `omr-cache` (mounted at `/vol`)
  - Cached dataset: `/vol/datasets/iter10_5_proxy_2k/`
  - 5-min rolling checkpoints: `/vol/ckpts/exp3_vir_vico/baseline/` and `/vol/ckpts/exp3_vir_vico/vir_vico/`
- **Modal Secret:** `hf-secret` (`HF_TOKEN` for private HF dataset hub)
- **Hugging Face Repo:** `gregoreesmaa/sheet-music-converter-omr` (`repo_type="dataset"`)
  - Eval comparison JSON: `evals/2026-11-26T00-54-ablation-vir-vico-router.json`
  - Run ledger JSON: `runs/2026-11-26T00-54-ablation-vir-vico-router.json`
  - Best EMA checkpoints: `checkpoints/2026-11-26T00-54-ablation-vir-vico-router/{baseline,vir_vico}/uni_omr_v2_ema_best.pt`

### Copy-Paste Monitoring & Control Commands (PowerShell)
```powershell
# Check Modal app status
.\.venv\Scripts\modal.exe app list

# Tail live logs
.\.venv\Scripts\modal.exe app logs ap-s2HoXUXOCRrpAI3CMHFQJF

# Stop run if needed (5-min checkpoints remain on Volume `omr-cache`)
.\.venv\Scripts\modal.exe app stop ap-s2HoXUXOCRrpAI3CMHFQJF

# Pull completed comparison JSON from Hugging Face
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; p = hf_hub_download('gregoreesmaa/sheet-music-converter-omr', 'evals/2026-11-26T00-54-ablation-vir-vico-router.json', repo_type='dataset'); print(open(p).read())"
```

## 3. Required Training & Eval Data
- **Dataset Shard:** `datasets/iter10_5_proxy_2k.tar` on `gregoreesmaa/sheet-music-converter-omr`
  - **Size:** `4,445,429,760` bytes (`4.14 GiB`)
  - **SHA-256:** `28853a11c3ee195c7ceb859ac2efcb03a445fb58077aebc82347215f48530aff`
  - **Splits:** `proxy_train.jsonl` (~2,000 stratified full pages: printed + synth HW + real HW) and `proxy_val.jsonl` (~60 held-out full pages).
  - **License:** Permissive only (zero `CC-BY-NC`, zero `suite31_hw` eval overlap).
- **Initialization:** Scratch `UniOMRv2Net` (`small_45m`, `seed=42`) — pure 1-variable isolation with zero warm-start bias toward `InkTokenRouterV2`.

## 4. Exact Launch Command & Hyperparameters
```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\modal.exe run --detach experiments/2026-11-26T00-54-ablation-vir-vico-router/run_modal.py
```

| Hyperparameter | Arm 1 (`baseline`) | Arm 2 (`vir_vico`) |
|---|---|---|
| Preset | `small_45m` (`d_model=384`, 2 axial + 4 global + 8 GQA dec, MTP-4) | Identical |
| Router | `InkTokenRouterV2` (`keep_ratio=0.40`, `n_bg_summary=4`) | `ViRTwoTierRouter` (`keep_ratio=0.40`, `k_coarse=96`, `static_buckets=True`) |
| ViCO Consistency Loss | Off (`0.0`) | On (`weight=0.10`, `temp=1.5`, every 4 steps vs dense `0.85` teacher) |
| Steps | `1,200` | `1,200` |
| Effective Batch Size | `8` (`batch_size=2` × `grad_accum=4`) | `8` (`batch_size=2` × `grad_accum=4`) |
| Resolution & Seq Len | `fullpage_max_dim=1280`, `max_seq_len=1536`, `multi_scale=True` | Identical |
| Optimizer & Schedule | `Muon` (2D Transformer) + fused `AdamW`, WSD `lr=3e-4`, warmup `120`, decay `25%` | Identical |
| Seed & Batch Order | `seed=42` (re-seeded pre-dataloader for bit-identical batches) | Identical |
| Eval Cadence | Steps `600` & `1,200` (raw + EMA: teacher-forced val loss/CE/MTP + `proxy_val.jsonl` SER/F1) | Identical |

## 5. 5-Minute Checkpointing & Resume Plan
- **Rolling Checkpoints:** Every `5.0` minutes, each arm atomically writes `uni_omr_v2_latest.pt` + `uni_omr_v2_ema_latest.pt` via `.tmp` → `os.replace` to `/vol/ckpts/exp3_vir_vico/<arm>/` and calls `vol.commit()`.
- **Best Checkpoints:** Saved on every new best `proxy_val.jsonl` SER (`uni_omr_v2_best.pt`, `uni_omr_v2_ema_best.pt`) and uploaded to HF on completion.
- **Resume Command:** Re-running the exact same `modal run --detach` command automatically detects `/vol/ckpts/exp3_vir_vico/<arm>/uni_omr_v2_latest.pt` and resumes from `saved_step + 1`.

## 6. Evaluation Gate & Decision Tree
- **Primary Metrics (at Step 1,200 EMA on `proxy_val.jsonl`):**
  1. Teacher-forced `val_loss`, `val_ce`, `val_mtp`
  2. Autoregressive `SER` (lower is better), `token_f1` (higher is better), `sys_count_exact`, `ink_recall`, `tokens_per_sec`, `mean_compression_x`
- **Decision Tree:**
  - **PROMOTE:** `vir_vico` beats `baseline` on `val_ce` AND (`SER` improves by `≥ 0.005` OR `token_f1` improves by `≥ 0.005` with ≤ 15% step-time overhead) → wire `ViRTwoTierRouter` + ViCO as the default router in `UniOMR v2` Iter-11 training!
  - **ITERATE:** `vir_vico` improves `val_ce`/`ink_recall` or `sys_count_exact` but lags slightly on `SER` or speed → inspect `k_coarse` (e.g. `64` vs `96`) and `vico_weight` (`0.05` vs `0.10`) in a targeted follow-up.
  - **DISCARD:** `vir_vico` degrades both `val_ce` and `SER`/`token_f1` relative to `InkTokenRouterV2` → keep `InkTokenRouterV2` for Iter-11 full training and document root cause in `OUTCOME.md`.
