# Experiment Plan — `2026-11-26T00-54-ablation-muon-vs-adamw`

## 1. Platform Chosen & Rationale
- **Platform:** **Modal `L4` (`24 GB` VRAM, Ada Lovelace `cc=8.9`, native BF16)** — `~$0.80/hr` out of `$30/mo` free Starter credit.
- **Why Modal `L4` (not Kaggle `T4`):** `UniOMR v2` training runs in native BF16 and uses PyTorch `>=2.10` native `torch.optim.Muon`. Turing `T4` (`cc=7.5`) emulates BF16 slowly; `L4` runs native BF16 + Flash/CUTLASS SDPA cleanly with zero local RTX 4080 SUPER contention.
- **Estimated Wall-Clock & Cost:** `~35–55 min` total for both `1,200`-step arms + `64`-page `proxy_val.jsonl` evals (`~$0.50–$0.75`). Function timeout set to `14400s` (4h) for safe headroom.

## 2. Identifiers for Follow-Up Queries (Zero-Prior-Context Handoff)
- **Modal Profile:** `gregoreesmaa1`
- **Modal App Name:** `exp4-ablation-muon-vs-adamw`
- **Modal App ID:** `ap-tXKsJnindxtL8tMh5RqEt9` (function call `fc-01M8AF527AAWGQAAMD5A90PSHT`)
- **Modal Dashboard URL:** https://modal.com/apps/gregoreesmaa1/main/ap-tXKsJnindxtL8tMh5RqEt9
- **Modal GPU / Runtime Verified:** `NVIDIA L4 (cc=8.9)` | `torch=2.15.0+cu132` | `native_muon=True`
- **Modal Volume:** `omr-cache` mounted at `/vol`
  - Cached dataset: `/vol/datasets/iter10_5_proxy_2k/`
  - Checkpoints & arm logs: `/vol/ckpts/2026-11-26T00-54-ablation-muon-vs-adamw/{muon,adamw}/`
- **Modal Secret:** `hf-secret` (`HF_TOKEN`)
- **Hugging Face Repo (`dataset`):** `gregoreesmaa/sheet-music-converter-omr`
  - Eval JSON: `evals/2026-11-26T00-54-ablation-muon-vs-adamw.json`
  - Run ledger: `runs/2026-11-26T00-54-ablation-muon-vs-adamw.json`
  - EMA checkpoints: `checkpoints/2026-11-26T00-54-ablation-muon-vs-adamw/{muon,adamw}_ema_latest.pt`

### Copy-Paste Monitoring / Retrieval Commands
```powershell
# Check live Modal app status
.\.venv\Scripts\modal.exe app list

# Stream or inspect logs
.\.venv\Scripts\modal.exe app logs ap-tXKsJnindxtL8tMh5RqEt9

# Stop runaway job if needed
.\.venv\Scripts\modal.exe app stop ap-tXKsJnindxtL8tMh5RqEt9

# Pull final comparison JSON from Hugging Face once COMPLETED
.\.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('gregoreesmaa/sheet-music-converter-omr', 'evals/2026-11-26T00-54-ablation-muon-vs-adamw.json', repo_type='dataset'))"
```

## 3. Required Training & Eval Data (100% Permissive, 0 Eval Overlap)
- **HF Shard:** `datasets/iter10_5_proxy_2k.tar` (`gregoreesmaa/sheet-music-converter-omr`)
  - `proxy_train.jsonl`: `2,000` deduplicated full-page OMR samples (stratified printed + synth HW + real HW; 0 `CC-BY-NC`, 0 `suite31_hw` overlap).
  - `proxy_val.jsonl`: `64` held-out full-page validation samples.
- **Warm-Start:** None (from-scratch `seed=42` initialization for clean 1-variable optimizer isolation).

## 4. Exact Launch Command & Hyperparameters (Commit `9d73170`)
```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\modal.exe run --detach experiments/2026-11-26T00-54-ablation-muon-vs-adamw/run_modal.py
```

Identical `train_uni_v2.py` CLI flags for both arms (ONLY `--optimizer` differs):
- `--preset small_45m`
- `--optimizer muon` vs `--optimizer adamw` *(isolated variable; runtime asserts `optimizer_mode == "muon+adamw"` vs `"adamw"`)*
- `--steps 1200 --seed 42`
- `--batch-size 2 --grad-accum 4` (effective batch = `8` full pages/step)
- `--curriculum fullpage --fullpage-manifest /vol/datasets/iter10_5_proxy_2k/proxy_train_abs.jsonl --stratified-fullpage`
- `--fullpage-max-dim 1280 --max-seq-len 2048 --multi-scale`
- `--augment --aug-prob 0.50`
- `--lr-schedule wsd --lr 3e-4 --warmup-steps 120 --wsd-decay-frac 0.25 --min-lr-ratio 0.05`
- `--grad-checkpoint --channels-last --bf16`
- `--gpu-ema-decay 0.997 --gpu-ema-every 5`
- `--save-every-minutes 5.0 --val-count 0 --eval-every 0 --log-every 50 --num-workers 0 --prefetch-threads 4`

## 5. 5-Minute Checkpointing & Resume Plan
- `--save-every-minutes 5.0` atomically writes `uni_omr_v2_latest.pt` and `uni_omr_v2_ema_latest.pt` via `.tmp` → `os.replace` into `/vol/ckpts/2026-11-26T00-54-ablation-muon-vs-adamw/<arm>/` and immediately calls `vol.commit()`.
- Completed arms persist `arm_result.json` to `/vol/ckpts/.../<arm>/arm_result.json`; re-running `run_modal.py` automatically skips any already-completed arm and resumes cleanly.

## 6. Evaluation Gate & Decision Tree
- **Metrics Measured:**
  - Per-100-step `loss`, `ce_loss`, `mtp_loss`, `stroke_hog_loss`, `grad_norm` (mean & p95), and `step_time_ms` / `steps_per_sec`.
  - Final `proxy_val.jsonl` (64 pages) teacher-forced **Val CE**, **Val MTP**, **Val Total Loss**, plus KV-cached greedy **Val SER**, **Token F1**, and **Sys-Count Exact** (both Raw and EMA checkpoints).
  - Peak VRAM (`alloc` & `reserved`) and wall-clock speed.
- **Promote (`Muon + AdamW`):** If `muon` beats `adamw` on EMA `proxy_val` CE (or matches within `<= 0.01` with lower greedy SER) AND wall-clock overhead is `<= 15%` with zero NaN/instability → lock `--optimizer muon` as the default for full Iter-11 `UniOMR v2` training.
- **Iterate:** If `muon` converges faster early but shows late-stage grad-norm spikes or slightly higher val CE → run a 2-point `--muon-lr` (`1.5e-4` vs `6e-4`) follow-up sweep.
- **Discard (`Muon`):** If pure fused `AdamW` wins on both EMA `proxy_val` CE and greedy SER, or `Muon` causes instability/excessive overhead → switch Iter-11 default to `--optimizer adamw`.
