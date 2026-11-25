# Experiment Plan — `2026-11-26T01-38-ov-27-mtp-k-weight-sweep`

> **Zero-Prior-Context Handoff:** Everything required to run, monitor, stop, resume, or evaluate Overnight Exp 27 (MTP-k × MTP Loss Weight Sweep) is self-contained in this folder (`README.md`, `PLAN.md`, `run_colab.py`).

## 1. Platform Chosen & Rationale
- **Provider:** Google Colab (`google-colab-cli` via `.\.venv\Scripts\colab.exe`).
- **Hardware:** `1× NVIDIA L4` (`24 GB` VRAM, Ada Lovelace `cc=8.9`, **native BF16**, `torch 2.14.1+cu130` pre-installed).
- **Session Name:** `ov-27-mtp-k-weight-sweep`
- **Estimated Wall-Clock & Cost:** ~36–40 minutes total (~3.2 min/arm × 9 arms + ~8 min `proxy_val` & speculative decode benchmark) ≈ `~3.0–3.2 CU`.
- **Rationale:** Native BF16, zero local GPU usage (`$env:CUDA_VISIBLE_DEVICES=""`), fast cold start, and clean teardown via `colab stop` in `finally`.

## 2. Identifiers for Follow-Up Queries
- **Colab Session:** `ov-27-mtp-k-weight-sweep`
- **Hugging Face Repo:** `gregoreesmaa/sheet-music-converter-omr` (`repo_type="dataset"`)
  - Full Eval JSON: `evals/2026-11-26T01-38-ov-27-mtp-k-weight-sweep.json`
  - Run Ledger JSON: `runs/2026-11-26T01-38-ov-27-mtp-k-weight-sweep.json`
- **Remote Paths on Colab VM:**
  - Code bundle (optional if pre-uploaded): `/content/bundle.tar.gz` → `/workspace/`
  - Datasets & rolling ckpts: `/content/vol/`
  - Transient token file: `/tmp/.hf_token` (`chmod 600`, deleted on read and in `finally`)

### Copy-Paste Monitoring & Control Commands (PowerShell)
```powershell
.\.venv\Scripts\colab.exe sessions
.\.venv\Scripts\colab.exe usage
.\.venv\Scripts\colab.exe stop -s ov-27-mtp-k-weight-sweep

$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; p = hf_hub_download('gregoreesmaa/sheet-music-converter-omr', 'evals/2026-11-26T01-38-ov-27-mtp-k-weight-sweep.json', repo_type='dataset'); print(open(p).read())"
```

## 3. Required Training & Eval Data
- **Supervised Proxy Dataset:** `datasets/iter10_5_proxy_2k.tar` (`2,000` train + `64` val full pages, permissive only, 0 `suite31_hw` overlap).
- **Initialization:** Scratch `UniOMRv2Net(UniOMRv2Config.small_45m(grad_checkpoint=True, mtp_k=k))` with identical `seed=42` across all 9 arms.

## 4. Exact Launch Command & Hyperparameters
```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\colab.exe new -s ov-27-mtp-k-weight-sweep --gpu L4
.\.venv\Scripts\colab.exe upload -s ov-27-mtp-k-weight-sweep <local_bundle.tar.gz> /content/bundle.tar.gz
.\.venv\Scripts\colab.exe upload -s ov-27-mtp-k-weight-sweep <local_tmp_token> /tmp/.hf_token
.\.venv\Scripts\colab.exe exec -s ov-27-mtp-k-weight-sweep -f experiments/2026-11-26T01-38-ov-27-mtp-k-weight-sweep/run_colab.py
.\.venv\Scripts\colab.exe stop -s ov-27-mtp-k-weight-sweep
```

| Grid Dimension | Values |
|---|---|
| `mtp_k` | `2`, `4`, `6` |
| `mtp_loss_weight` | `0.15`, `0.30`, `0.50` |
| Total Arms | `9` (`mtp2_w0.15` .. `mtp6_w0.50`) |
| Steps / Eff Batch | `300` / `8` (`B=2 × ga=4`) |
| Resolution & Seq | `fullpage_max_dim=1280`, `max_seq_len=1536` |
| Precision | Native `BF16` |
| Optimizer & Schedule | Fused `AdamW` (`lr=3e-4`, `wd=0.04`), WSD (`warmup=40`, `decay=25%`) |

## 5. 5-Minute Checkpointing & Resume Plan
- Every `5.0` minutes during each arm, `run_colab.py` atomically writes `uni_omr_v2_latest.pt` + `uni_omr_v2_ema_latest.pt` (`.tmp` → `os.replace`) to `/content/vol/ckpts/2026-11-26T01-38-ov-27-mtp-k-weight-sweep/<arm>/`.
- Completed arm JSONs are cached in `arm_result.json` so re-execution resumes cleanly.

## 6. Evaluation Gate & Decision Tree
- **Primary Metrics:**
  1. `proxy_val` (64 pages): `val_token_f1`, `val_ser`, `val_ce_loss`, `val_mtp_loss`, `mtp_params`.
  2. **Lossless Self-Speculative Decoding (16 val pages, `B=1`, `max_len=384`):** `bit_exact_parity` (asserted vs `model.generate()`), `draft_acceptance_rate`, `mean_accepted_tokens_per_step`, `greedy_tok_per_sec`, `spec_tok_per_sec`, `speedup_x`.
- **Decision Tree:**
  - **PROMOTE:** Winning `(mtp_k, mtp_loss_weight)` sits on the Pareto frontier (highest `val_token_f1` or within `0.003` with `>= 1.35×` lossless speculative speedup) → adopt as Iter-11 `UniOMR v2` default and promote lossless `generate_speculative()` into `uni_omr_v2.py`.
  - **ITERATE:** High `mtp_loss_weight=0.50` boosts acceptance rate dramatically but slightly hurts `val_ce_loss` → test cosine-decaying `mtp_loss_weight` (0.50 → 0.15).
  - **DISCARD:** `mtp_k=4, w=0.30` remains Pareto-optimal → keep current `UniOMR v2` defaults unchanged.
