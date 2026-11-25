# Experiment Plan — `2026-11-26T01-38-ov-40-small45m-vs-edge85m-pareto`

> **Zero-Prior-Context Handoff:** Everything needed to run, monitor, stop, resume, or evaluate Overnight Exp 40 (`small_45m` vs `edge85m` Pareto curve) is self-contained in this folder (`README.md`, `PLAN.md`, `run_colab.py`).

## 1. Platform Chosen & Rationale
- **Provider:** Google Colab (`google-colab-cli` via `.\.venv\Scripts\colab.exe`).
- **Hardware:** `1x NVIDIA L4` (`24 GB` VRAM, Ada Lovelace `cc=8.9`, **native BF16**, `torch 2.14.1+cu130` pre-installed).
- **Session Name:** `ov-40-pareto`
- **Estimated Wall-Clock & Cost:** ~35–42 minutes total (~10 min `small_45m` + ~16 min `edge85m` + ~10–14 min Rust build, `proxy_val`, and 45-piece `suite31_hw` evals) ≈ `~3.0–3.5 CU`.
- **Rationale:** Native BF16 with 24 GB VRAM comfortably fits both `small_45m` and `edge85m` at `fullpage_max_dim=1280` (`grad_checkpoint=True`, `B=2`, `grad_accum=4`) with zero local GPU usage.

## 2. Identifiers for Follow-Up Queries
- **Colab Session:** `ov-40-pareto`
- **Hugging Face Repo:** `gregoreesmaa/sheet-music-converter-omr` (`repo_type="dataset"`)
  - Full Eval JSON: `evals/2026-11-26T01-38-ov-40-small45m-vs-edge85m-pareto.json`
  - Run Ledger JSON: `runs/2026-11-26T01-38-ov-40-small45m-vs-edge85m-pareto.json`
- **Remote Paths on Colab VM:**
  - Code bundle: `/content/ov_bundle.tar.gz` → extracted to `/workspace/`
  - Datasets & rolling ckpts: `/content/vol/`
  - Transient token file: `/tmp/.hf_token` (`chmod 600`, purged on read and in `finally`)

### Copy-Paste Monitoring & Control Commands (PowerShell)
```powershell
# Check active Colab sessions & compute-unit balance
.\.venv\Scripts\colab.exe sessions
.\.venv\Scripts\colab.exe usage

# Stop Colab session immediately (ALWAYS run in finally!)
.\.venv\Scripts\colab.exe stop -s ov-40-pareto

# Fetch completed eval JSON from Hugging Face
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; p = hf_hub_download('gregoreesmaa/sheet-music-converter-omr', 'evals/2026-11-26T01-38-ov-40-small45m-vs-edge85m-pareto.json', repo_type='dataset'); print(open(p).read())"
```

## 3. Required Training & Eval Data
- **Supervised Proxy Dataset:** `datasets/iter10_5_proxy_2k.tar` (`2,000` train + `64` val full pages, permissive only, 0 `suite31_hw` overlap).
- **OOD Evaluation Benchmark:** `datasets/suite31_hw_eval.tar` (45 pieces: 31 printed + 14 real historical handwritten manuscripts, `clean` + `photo` = 90 pages).
- **Initialization:** Scratch `UniOMRv2Net` with identical `seed=42` and identical shuffled batch stream across both presets.

## 4. Exact Launch Command & Hyperparameters
```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\colab.exe new -s ov-40-pareto --gpu L4
.\.venv\Scripts\colab.exe upload -s ov-40-pareto <local_bundle.tar.gz> /content/ov_bundle.tar.gz
.\.venv\Scripts\colab.exe upload -s ov-40-pareto <local_tmp_token> /tmp/.hf_token
.\.venv\Scripts\colab.exe exec -s ov-40-pareto -f experiments/2026-11-26T01-38-ov-40-small45m-vs-edge85m-pareto/run_colab.py
.\.venv\Scripts\colab.exe stop -s ov-40-pareto
```

| Hyperparameter | Arm A (`small_45m`) | Arm B (`edge85m`) |
|---|---|---|
| Preset | `UniOMRv2Config.small_45m(grad_checkpoint=True)` | `UniOMRv2Config.edge85m(grad_checkpoint=True)` |
| Est. WebGPU Size (`block_size=32`) | `~31 MB` Hybrid / `~23 MB` INT4 (`< 55 MB`) | `~52 MB` Hybrid / `~43 MB` INT4 (`< 55 MB`) |
| Steps | `1,000` | `1,000` |
| Effective Batch Size | `8` (`batch_size=2` × `grad_accum=4`) | `8` (`batch_size=2` × `grad_accum=4`) |
| Resolution & Seq Len | `fullpage_max_dim=1280`, `max_seq_len=1536` | Identical |
| Precision | Native `BF16` (`CUTLASS` SDPA; `cudnn_sdp=False`) | Identical |
| Optimizer & Schedule | Fused `AdamW` (`lr=3e-4`, `wd=0.04`), WSD (`warmup=100`, `decay=25%`, `min_lr=0.05`) | Identical |
| EMA & Seed | `GPUEmaShadow(decay=0.995)`, `seed=42` | Identical |

## 5. 5-Minute Checkpointing & Resume Plan
- Every `5.0` minutes during each arm, `run_colab.py` atomically writes `uni_omr_v2_latest.pt` + `uni_omr_v2_ema_latest.pt` (`.tmp` → `os.replace`) to `/content/vol/ckpts/<exp_id>/<arm>/`.
- Completed arm results are cached to `/content/vol/ckpts/<exp_id>/<arm>/arm_result.json` so re-executing `run_colab.py` skips completed arms cleanly.

## 6. Evaluation Gate & Decision Tree
- **Primary Metrics:**
  1. Model size gate: `hybrid_fp16_int4_with_scales_mb < 55.0` for both presets.
  2. `proxy_val` (64 pages): `val_ce_loss`, `val_token_f1`, `val_ser`, `val_sys_count_exact`, inference `ms/page`.
  3. `suite31_hw` (45 pieces × 2 domains = 90 pages): `printed_clean`, `printed_photo`, `hw_clean`, `hw_photo`, `all45_clean`, `all45_photo` (`tok_f1`, `note_f1`, `decode_ok_rate`).
  4. Pareto efficiency: `delta_all45_note_f1`, `delta_val_tok_f1`, `step_ms_ratio`, `infer_ms_ratio`, `size_delta_mb`.
- **Decision Tree:**
  - **PROMOTE `edge85m`:** `edge85m` stays `< 55 MB` Hybrid AND improves `proxy_val` Token F1 by `>= +0.020` or `suite31_hw` `all45` Token/Note F1 by `>= +0.015` with `< 2.0x` inference latency overhead → select `edge85m` for Iter-11 flagship `A100/H100` run.
  - **ITERATE:** `edge85m` beats `small_45m` on `val_ce_loss` but underfits at `1,000` steps → run longer `3,000`-step comparison or warm-start from `I-JEPA` encoder.
  - **KEEP `small_45m`:** `small_45m` matches or beats `edge85m` within `0.010` F1 at `~20 MB` smaller footprint and faster inference → keep `small_45m` as primary WebGPU target.
