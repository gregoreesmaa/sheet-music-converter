# Experiment Plan — `2026-11-26T01-38-ov-25-cass-prob-sweep`

> **Zero-Prior-Context Handoff:** Everything required to run, monitor, stop, resume, or evaluate Overnight Exp 25 (CASS Scheduled Sampling Sweep) is self-contained in this folder (`README.md`, `PLAN.md`, `run_colab.py`).

## 1. Platform Chosen & Rationale
- **Provider:** Google Colab (`google-colab-cli` via `.\.venv\Scripts\colab.exe`).
- **Hardware:** `1× NVIDIA L4` (`24 GB` VRAM, Ada Lovelace `cc=8.9`, **native BF16**, `torch 2.14.1+cu130` pre-installed).
- **Session Name:** `ov-25-cass-prob-sweep`
- **Estimated Wall-Clock & Cost:** ~32–36 minutes total (~7.5 min for `cass=0.0`, ~9.5 min each for `cass=0.10`/`0.25`, ~5 min for `proxy_val` long-page evaluation) ≈ `~2.6–2.9 CU`.
- **Rationale:** Native BF16, zero local GPU usage (`$env:CUDA_VISIBLE_DEVICES=""`), fast cold start, and clean teardown via `colab stop` in `finally`.

## 2. Identifiers for Follow-Up Queries
- **Colab Session:** `ov-25-cass-prob-sweep`
- **Hugging Face Repo:** `gregoreesmaa/sheet-music-converter-omr` (`repo_type="dataset"`)
  - Full Eval JSON: `evals/2026-11-26T01-38-ov-25-cass-prob-sweep.json`
  - Run Ledger JSON: `runs/2026-11-26T01-38-ov-25-cass-prob-sweep.json`
- **Remote Paths on Colab VM:**
  - Code bundle (optional if pre-uploaded): `/content/bundle.tar.gz` → `/workspace/`
  - Datasets & rolling ckpts: `/content/vol/`
  - Transient token file: `/tmp/.hf_token` (`chmod 600`, deleted on read and in `finally`)

### Copy-Paste Monitoring & Control Commands (PowerShell)
```powershell
.\.venv\Scripts\colab.exe sessions
.\.venv\Scripts\colab.exe usage
.\.venv\Scripts\colab.exe stop -s ov-25-cass-prob-sweep

$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; p = hf_hub_download('gregoreesmaa/sheet-music-converter-omr', 'evals/2026-11-26T01-38-ov-25-cass-prob-sweep.json', repo_type='dataset'); print(open(p).read())"
```

## 3. Required Training & Eval Data
- **Supervised Proxy Dataset:** `datasets/iter10_5_proxy_2k.tar` (`2,000` train + `64` val full pages, permissive only, 0 `suite31_hw` overlap).
- **Initialization:** Scratch `UniOMRv2Net(UniOMRv2Config.small_45m(grad_checkpoint=True, tok_dropout=0.0))` with identical `seed=42` across all 3 arms.

## 4. Exact Launch Command & Hyperparameters
```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\colab.exe new -s ov-25-cass-prob-sweep --gpu L4
.\.venv\Scripts\colab.exe upload -s ov-25-cass-prob-sweep <local_bundle.tar.gz> /content/bundle.tar.gz
.\.venv\Scripts\colab.exe upload -s ov-25-cass-prob-sweep <local_tmp_token> /tmp/.hf_token
.\.venv\Scripts\colab.exe exec -s ov-25-cass-prob-sweep -f experiments/2026-11-26T01-38-ov-25-cass-prob-sweep/run_colab.py
.\.venv\Scripts\colab.exe stop -s ov-25-cass-prob-sweep
```

| Hyperparameter | Arm 1 (`cass_0.00`) | Arm 2 (`cass_0.10`) | Arm 3 (`cass_0.25`) |
|---|---|---|---|
| `cass_prob` | `0.0` | `0.10` | `0.25` |
| `tok_dropout` | `0.0` | `0.0` | `0.0` |
| Architecture | `UniOMR v2 small_45m` | Identical | Identical |
| Steps / Eff Batch | `700` / `8` (`B=2 × ga=4`) | Identical | Identical |
| Resolution & Seq | `fullpage_max_dim=1280`, `max_seq_len=1536` | Identical | Identical |
| Precision | Native `BF16` | Identical | Identical |
| Optimizer & Schedule | Fused `AdamW` (`lr=3e-4`, `wd=0.04`), WSD (`warmup=70`, `decay=25%`) | Identical | Identical |
| Eval Max Len | `1024` (full long-page horizon) | Identical | Identical |

## 5. 5-Minute Checkpointing & Resume Plan
- Every `5.0` minutes during each arm, `run_colab.py` atomically writes `uni_omr_v2_latest.pt` + `uni_omr_v2_ema_latest.pt` (`.tmp` → `os.replace`) to `/content/vol/ckpts/2026-11-26T01-38-ov-25-cass-prob-sweep/<arm>/`.
- Completed arm JSONs are cached in `/content/vol/ckpts/2026-11-26T01-38-ov-25-cass-prob-sweep/<arm>/arm_result.json` so re-execution resumes seamlessly.

## 6. Evaluation Gate & Decision Tree
- **Primary Metrics (`proxy_val`, 64 pages, `eval_max_len=1024`):**
  1. Overall `val_token_f1`, `val_ser`, `val_ce_loss`, `val_sys_count_exact`.
  2. **Long-page exposure-bias metrics:** length-bucketed Token F1 & SER (`short <= 256`, `medium 257..512`, `long > 512`), `mean_len_ratio` (`pred_len / ref_len`), and `early_eos_rate` (`pred_len < 0.75 * ref_len`).
  3. Throughput overhead (`step_ms_mean`).
- **Decision Tree:**
  - **PROMOTE:** Winning `cass_prob > 0` improves overall `val_token_f1` by `>= +0.005` AND improves `long > 512` Token F1 with `<= 30%` step-time overhead → enable winning `--cass-prob` in Iter-11 `UniOMR v2` training.
  - **ITERATE:** `long > 512` improves, but early-training CASS slows convergence from scratch → test delayed CASS curriculum (enable CASS only in final 40% of steps).
  - **DISCARD:** `cass_prob=0.0` beats both `0.10` and `0.25` on both overall and long-page F1 → keep `cass_prob=0.0` for Iter-11.
