# Experiment Plan — `2026-11-26T01-38-ov-19-router-saliency-keep-sweep`

> **Zero-Prior-Context Handoff:** This folder (`README.md`, `PLAN.md`, `run_colab.py`) is 100% self-contained. Any agent can launch, monitor, stop, resume, or evaluate Overnight Exp 19 without prior conversation context.

## 1. Platform Chosen & Rationale
- **Provider:** Google Colab (`.\.venv\Scripts\colab.exe`, Pro+ compute-unit pool).
- **Hardware:** `1× NVIDIA L4` (`24 GB` VRAM, Ada Lovelace `cc=8.9`, native `BF16`, `torch 2.14.1+cu130` pre-installed).
- **Session Name:** `ov-19-router-sweep`
- **Estimated Wall-Clock & Cost:** ~30–35 minutes total (~8.5 min per 600-step arm × 3 + ~5 min val generation & cross-threshold matrix) ≈ `~2.5–3.0 CU`.
- **Rationale:** Native `BF16` matches our target training precision, zero local GPU usage (`$env:CUDA_VISIBLE_DEVICES=""`), and fast ~8s cold start with `torch` pre-installed.

## 2. Identifiers for Follow-Up Queries
- **Colab Session:** `ov-19-router-sweep`
- **Hugging Face Repo:** `gregoreesmaa/sheet-music-converter-omr` (`repo_type="dataset"`)
  - Full Eval JSON: `evals/2026-11-26T01-38-ov-19-router-saliency-keep-sweep.json`
  - Run Ledger JSON: `runs/2026-11-26T01-38-ov-19-router-saliency-keep-sweep.json`
- **Remote Paths on Colab VM:**
  - Code bundle (optional if session upload): `/content/bundle.tar.gz` → `/workspace/`
  - Datasets & rolling checkpoints: `/content/vol/`
  - Transient token file: `/tmp/.hf_token` (`chmod 600`, deleted immediately on read and in `finally`)

### Copy-Paste Monitoring & Control Commands (PowerShell)
```powershell
# Check active Colab sessions & compute-unit balance
.\.venv\Scripts\colab.exe sessions
.\.venv\Scripts\colab.exe usage

# Stop Colab session immediately (ALWAYS run in finally!)
.\.venv\Scripts\colab.exe stop -s ov-19-router-sweep

# Fetch completed eval JSON from Hugging Face
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; p = hf_hub_download('gregoreesmaa/sheet-music-converter-omr', 'evals/2026-11-26T01-38-ov-19-router-saliency-keep-sweep.json', repo_type='dataset'); print(open(p).read())"
```

## 3. Required Training & Eval Data
- **Supervised Proxy Dataset:** `gregoreesmaa/sheet-music-converter-omr` → `datasets/iter10_5_proxy_2k.tar` (`2,000` train + `64` val full pages, permissive only, zero `CC-BY-NC`, zero `suite31_hw` overlap).
- **Initialization:** Scratch `UniOMRv2Net(UniOMRv2Config.small_45m(...))` with identical `seed=42` across all 3 arms.

## 4. Exact Launch Command & Hyperparameters
```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\colab.exe new -s ov-19-router-sweep --gpu L4
.\.venv\Scripts\colab.exe upload -s ov-19-router-sweep <local_bundle.tar.gz> /content/bundle.tar.gz
.\.venv\Scripts\colab.exe upload -s ov-19-router-sweep <local_tmp_token> /tmp/.hf_token
.\.venv\Scripts\colab.exe exec -s ov-19-router-sweep -f experiments/2026-11-26T01-38-ov-19-router-saliency-keep-sweep/run_colab.py
.\.venv\Scripts\colab.exe stop -s ov-19-router-sweep
```

| Hyperparameter | `arm_003_055` (Conservative) | `arm_005_045` (Balanced) | `arm_008_035` (Aggressive) |
|---|---:|---:|---:|
| `saliency_thr` | `0.03` | `0.05` | `0.08` |
| `max_keep_ratio` | `0.55` | `0.45` | `0.35` |
| `min_keep_ratio` / `keep_ratio` | `0.15` / `0.40` | `0.15` / `0.40` | `0.15` / `0.40` |
| `dynamic_routing` / `max_mem_tokens` | `True` / `3072` | `True` / `3072` | `True` / `3072` |
| Architecture | `UniOMR v2 small_45m` | `UniOMR v2 small_45m` | `UniOMR v2 small_45m` |
| Steps / Eff. Batch | `600` / `8` (`B=2 × GA=4`) | `600` / `8` (`B=2 × GA=4`) | `600` / `8` (`B=2 × GA=4`) |
| Resolution / `max_seq_len` | `1280` / `1536` | `1280` / `1536` | `1280` / `1536` |
| Optimizer / Schedule | `Muon + AdamW` (`lr=3e-4`, WSD) | Identical | Identical |
| Seed / Precision | `42` / Native `BF16` | `42` / Native `BF16` | `42` / Native `BF16` |

## 5. 5-Minute Checkpointing & Resume Plan
- Every `5.0` minutes during each arm, `run_colab.py` atomically writes `uni_omr_v2_latest.pt` + `uni_omr_v2_ema_latest.pt` (`.tmp` → `os.replace`) to `/content/vol/ckpts/2026-11-26T01-38-ov-19-router-saliency-keep-sweep/<arm>/`.
- Completed arms write `/content/vol/ckpts/.../<arm>/arm_result.json`; re-running `run_colab.py` in the same session skips finished arms and resumes in-progress arms from the latest 5-minute checkpoint.

## 6. Evaluation Gate & Decision Tree
- **Primary Metrics (on `proxy_val.jsonl`, 64 pages, EMA weights):**
  1. `ink_recall` (mean & p10 across val pages)
  2. `mean_compression_x` (`n_in / n_out`) & `keep_fraction` (`k_kept / n_in`)
  3. `val_token_f1`, `val_ser`, `val_ce_loss`, `val_sys_count_exact`, `tokens_per_sec`
- **Decision Tree:**
  - **PROMOTE:** Adopt the highest-compression `(saliency_thr, max_keep_ratio)` setting that maintains `ink_recall >= 0.96` AND `val_token_f1` within `<= 0.005` of (or better than) `conservative (0.03 / 0.55)` as the new `UniOMRv2Config` default for Iter-11.
  - **ITERATE:** If `balanced (0.05 / 0.45)` beats `conservative` on F1/speed while `aggressive (0.08 / 0.35)` drops ink recall `< 0.94`, test an intermediate `(0.06 / 0.40)` point on full-scale Iter-11.
  - **DISCARD:** If tightening `max_keep_ratio <= 0.55` degrades `val_token_f1` by `> 0.015` across all three arms, keep `max_keep_ratio=0.75` and shift compression to `ViRTwoTierRouter` (Exp 20).
