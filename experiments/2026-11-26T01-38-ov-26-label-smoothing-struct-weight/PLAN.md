# Experiment Plan — `2026-11-26T01-38-ov-26-label-smoothing-struct-weight`

> **Zero-Prior-Context Handoff:** Everything required to run, monitor, stop, resume, or evaluate Overnight Exp 26 (Label Smoothing × Structural Token Weight Sweep) is self-contained in this folder (`README.md`, `PLAN.md`, `run_colab.py`).

## 1. Platform Chosen & Rationale
- **Provider:** Google Colab (`google-colab-cli` via `.\.venv\Scripts\colab.exe`).
- **Hardware:** `1× NVIDIA L4` (`24 GB` VRAM, Ada Lovelace `cc=8.9`, **native BF16**, `torch 2.14.1+cu130` pre-installed).
- **Session Name:** `ov-26-ls-struct-weight`
- **Estimated Wall-Clock & Cost:** ~34–38 minutes total (~4.8 min/arm × 6 arms + ~6 min `proxy_val` eval across 6 arms) ≈ `~2.8–3.1 CU`.
- **Rationale:** Native BF16, zero local GPU usage (`$env:CUDA_VISIBLE_DEVICES=""`), fast cold start, and clean teardown via `colab stop` in `finally`.

## 2. Identifiers for Follow-Up Queries
- **Colab Session:** `ov-26-ls-struct-weight`
- **Hugging Face Repo:** `gregoreesmaa/sheet-music-converter-omr` (`repo_type="dataset"`)
  - Full Eval JSON: `evals/2026-11-26T01-38-ov-26-label-smoothing-struct-weight.json`
  - Run Ledger JSON: `runs/2026-11-26T01-38-ov-26-label-smoothing-struct-weight.json`
- **Remote Paths on Colab VM:**
  - Code bundle (optional if pre-uploaded): `/content/bundle.tar.gz` → `/workspace/`
  - Datasets & rolling ckpts: `/content/vol/`
  - Transient token file: `/tmp/.hf_token` (`chmod 600`, deleted on read and in `finally`)

### Copy-Paste Monitoring & Control Commands (PowerShell)
```powershell
.\.venv\Scripts\colab.exe sessions
.\.venv\Scripts\colab.exe usage
.\.venv\Scripts\colab.exe stop -s ov-26-ls-struct-weight

$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; p = hf_hub_download('gregoreesmaa/sheet-music-converter-omr', 'evals/2026-11-26T01-38-ov-26-label-smoothing-struct-weight.json', repo_type='dataset'); print(open(p).read())"
```

## 3. Required Training & Eval Data
- **Supervised Proxy Dataset:** `datasets/iter10_5_proxy_2k.tar` (`2,000` train + `64` val full pages, permissive only, 0 `suite31_hw` overlap).
- **Initialization:** Scratch `UniOMRv2Net(UniOMRv2Config.small_45m(grad_checkpoint=True))` with identical `seed=42` across all 6 arms.

## 4. Exact Launch Command & Hyperparameters
```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\colab.exe new -s ov-26-ls-struct-weight --gpu L4
.\.venv\Scripts\colab.exe upload -s ov-26-ls-struct-weight <local_bundle.tar.gz> /content/bundle.tar.gz
.\.venv\Scripts\colab.exe upload -s ov-26-ls-struct-weight <local_tmp_token> /tmp/.hf_token
.\.venv\Scripts\colab.exe exec -s ov-26-ls-struct-weight -f experiments/2026-11-26T01-38-ov-26-label-smoothing-struct-weight/run_colab.py
.\.venv\Scripts\colab.exe stop -s ov-26-ls-struct-weight
```

| Grid Dimension | Values |
|---|---|
| `label_smoothing` | `0.00`, `0.05`, `0.10` |
| `struct_tok_weight` | `1.00`, `1.25` |
| Total Arms | `6` (`ls0.00_sw1.00`, `ls0.00_sw1.25`, `ls0.05_sw1.00`, `ls0.05_sw1.25`, `ls0.10_sw1.00`, `ls0.10_sw1.25`) |
| Steps / Eff Batch | `450` / `8` (`B=2 × ga=4`) |
| Resolution & Seq | `fullpage_max_dim=1280`, `max_seq_len=1536` |
| Precision | Native `BF16` |
| Optimizer & Schedule | Fused `AdamW` (`lr=3e-4`, `wd=0.04`), WSD (`warmup=50`, `decay=25%`) |

## 5. 5-Minute Checkpointing & Resume Plan
- Every `5.0` minutes during each arm, `run_colab.py` atomically writes `uni_omr_v2_latest.pt` + `uni_omr_v2_ema_latest.pt` (`.tmp` → `os.replace`) to `/content/vol/ckpts/2026-11-26T01-38-ov-26-label-smoothing-struct-weight/<arm>/`.
- Completed arm JSONs are cached in `arm_result.json` so re-execution resumes seamlessly without redoing finished arms.

## 6. Evaluation Gate & Decision Tree
- **Primary Metrics (`proxy_val`, 64 pages, apples-to-apples unweighted eval CE with `ls=0.0`):**
  1. Overall `val_token_f1`, `val_ser`, `val_ce_unweighted`, `val_sys_count_exact`.
  2. **Structural vs Content breakdown:** `struct_token_f1` (`clef-*`, `key*`, `time-*`, `<sys>`, `grace`, `bar*`) and `content_token_f1` (all remaining notation tokens).
  3. **Calibration:** `val_top1_acc`, `val_mean_conf`, and `val_ece_10bin`.
- **Decision Tree:**
  - **PROMOTE:** Winning `(label_smoothing, struct_tok_weight)` beats the `(0.05, 1.00)` current v2 default by `>= +0.005` on overall `val_token_f1` or `>= +0.015` on `struct_token_f1` without regressing `content_token_f1` → wire `struct_tok_weight` directly into `UniOMRv2Net.compute_loss()` and `train_uni_v2.py` CLI and set the winning pair as Iter-11 default.
  - **ITERATE:** `struct_tok_weight=1.25` improves `struct_token_f1` monotonically with zero content loss → test `struct_tok_weight=1.50` in a quick follow-up.
  - **DISCARD:** `(0.05, 1.00)` or `(0.00, 1.00)` wins cleanly → keep unweighted CE with winning `label_smoothing`.
