# Experiment Plan — `2026-11-26T01-38-ov-20-vir-vico-weight-schedule`

> **Zero-Prior-Context Handoff:** This folder (`README.md`, `PLAN.md`, `run_colab.py`) is 100% self-contained. Any agent can launch, monitor, stop, resume, or evaluate Overnight Exp 20 without prior conversation context.

## 1. Platform Chosen & Rationale
- **Provider:** Google Colab (`.\.venv\Scripts\colab.exe`, Pro+ compute-unit pool).
- **Hardware:** `1× NVIDIA L4` (`24 GB` VRAM, Ada Lovelace `cc=8.9`, native `BF16`, `torch 2.14.1+cu130` pre-installed).
- **Session Name:** `ov-20-vir-vico-sweep`
- **Estimated Wall-Clock & Cost:** ~32–38 minutes total (~7.0–7.5 min per 500-step arm × 4 + ~6 min val & dense-vs-routed KL evaluation) ≈ `~2.7–3.2 CU`.
- **Rationale:** Native `BF16` with `vico_every=2` dense teacher forward passes fits comfortably within 24 GB VRAM (`grad_checkpoint=True`, `B=2 × GA=4`), zero local GPU usage (`$env:CUDA_VISIBLE_DEVICES=""`).

## 2. Identifiers for Follow-Up Queries
- **Colab Session:** `ov-20-vir-vico-sweep`
- **Hugging Face Repo:** `gregoreesmaa/sheet-music-converter-omr` (`repo_type="dataset"`)
  - Full Eval JSON: `evals/2026-11-26T01-38-ov-20-vir-vico-weight-schedule.json`
  - Run Ledger JSON: `runs/2026-11-26T01-38-ov-20-vir-vico-weight-schedule.json`
- **Remote Paths on Colab VM:**
  - Code bundle (optional): `/content/bundle.tar.gz` → `/workspace/`
  - Datasets & rolling checkpoints: `/content/vol/`
  - Transient token file: `/tmp/.hf_token` (`chmod 600`, deleted immediately on read and in `finally`)

### Copy-Paste Monitoring & Control Commands (PowerShell)
```powershell
# Check active Colab sessions & compute-unit balance
.\.venv\Scripts\colab.exe sessions
.\.venv\Scripts\colab.exe usage

# Stop Colab session immediately (ALWAYS run in finally!)
.\.venv\Scripts\colab.exe stop -s ov-20-vir-vico-sweep

# Fetch completed eval JSON from Hugging Face
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; p = hf_hub_download('gregoreesmaa/sheet-music-converter-omr', 'evals/2026-11-26T01-38-ov-20-vir-vico-weight-schedule.json', repo_type='dataset'); print(open(p).read())"
```

## 3. Required Training & Eval Data
- **Supervised Proxy Dataset:** `gregoreesmaa/sheet-music-converter-omr` → `datasets/iter10_5_proxy_2k.tar` (`2,000` train + `64` val full pages, permissive only, zero `CC-BY-NC`, zero `suite31_hw` overlap).
- **Initialization:** Scratch `UniOMRv2Net(UniOMRv2Config.small_45m(...))` with `model.encoder.router = ViRTwoTierRouter(...)`, identical `seed=42` across all 4 arms.

## 4. Exact Launch Command & Hyperparameters
```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\colab.exe new -s ov-20-vir-vico-sweep --gpu L4
.\.venv\Scripts\colab.exe upload -s ov-20-vir-vico-sweep <local_bundle.tar.gz> /content/bundle.tar.gz
.\.venv\Scripts\colab.exe upload -s ov-20-vir-vico-sweep <local_tmp_token> /tmp/.hf_token
.\.venv\Scripts\colab.exe exec -s ov-20-vir-vico-sweep -f experiments/2026-11-26T01-38-ov-20-vir-vico-weight-schedule/run_colab.py
.\.venv\Scripts\colab.exe stop -s ov-20-vir-vico-sweep
```

| Hyperparameter | `vico_w002_kr035` | `vico_w008_kr035` | `vico_w020_kr035` | `vico_w008_kr025` |
|---|---:|---:|---:|---:|
| `vico_loss_weight` | `0.02` | `0.08` | `0.20` | `0.08` |
| `keep_ratio` (target compression) | `0.35` (~2.85×) | `0.35` (~2.85×) | `0.35` (~2.85×) | `0.25` (~4.0×) |
| `dense_teacher_keep_ratio` | `0.85` | `0.85` | `0.85` | `0.85` |
| `vico_every` / `vico_temp` | `2` / `1.5` | `2` / `1.5` | `2` / `1.5` | `2` / `1.5` |
| `ViRTwoTierRouter` (`k_coarse`, `static_buckets`) | `96`, `True` | `96`, `True` | `96`, `True` | `96`, `True` |
| Steps / Eff. Batch | `500` / `8` (`B=2 × GA=4`) | Identical | Identical | Identical |
| Resolution / `max_seq_len` | `1280` / `1536` | Identical | Identical | Identical |
| Optimizer / Schedule | `Muon + AdamW` (`lr=3e-4`, WSD) | Identical | Identical | Identical |
| Seed / Precision | `42` / Native `BF16` | `42` / Native `BF16` | `42` / Native `BF16` | `42` / Native `BF16` |

## 5. 5-Minute Checkpointing & Resume Plan
- Every `5.0` minutes during each arm, `run_colab.py` atomically writes `uni_omr_v2_latest.pt` + `uni_omr_v2_ema_latest.pt` (`.tmp` → `os.replace`) to `/content/vol/ckpts/2026-11-26T01-38-ov-20-vir-vico-weight-schedule/<arm>/`.
- Completed arms write `arm_result.json`; re-running `run_colab.py` in the same session skips finished arms and resumes in-progress arms cleanly.

## 6. Evaluation Gate & Decision Tree
- **Primary Metrics (on `proxy_val.jsonl`, 64 pages, EMA weights):**
  1. **ViCO Health Check:** `vico_step50_val > 0.0`, `vico_nonzero_frac >= 0.99` on active steps, `mean_k_fine_routed < mean_k_fine_dense`.
  2. **Dense-vs-Routed Alignment:** `eval_vico_kl_native` and `eval_vico_kl_kr025` (symmetric KL between routed student and `0.85` dense teacher).
  3. **Accuracy & Efficiency:** `val_token_f1`, `val_ser`, `val_ce`, `ink_recall_mean`, `compression_x_mean`, `tokens_per_sec`.
- **Decision Tree:**
  - **PROMOTE:** Adopt the winning `(vico_loss_weight, keep_ratio)` pair if it achieves lower `eval_vico_kl` AND higher `val_token_f1` than `vico_w002_kr035` with `>= 2.5×` compression and `vico_step50_val > 0.0`.
  - **ITERATE:** If `vico_w008_kr025` maintains `val_token_f1` within `0.005` of `kr035` at `~3.8×` compression, run a follow-up full `suite31_hw` evaluation with `k_coarse ∈ {64, 96, 128}`.
  - **DISCARD:** If higher `vico_loss_weight` monotonically degrades `val_ce` and `val_token_f1` while failing to reduce `eval_vico_kl`, keep `ViRTwoTierRouter` with `vico_loss_weight=0.0` (router-only) or stick with `InkTokenRouterV2`.
