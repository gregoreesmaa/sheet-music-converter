# Experiment Plan — `2026-11-26T01-30-academic-gauge-rope`

> **Zero-Prior-Context Handoff:** Everything required to run, monitor, stop, resume, or evaluate Academic Experiment B (Gauge-RoPE) is self-contained in this folder (`README.md`, `PLAN.md`, `run_colab.py`) plus `tmp-train/mvd/gauge_rope.py`.

## 1. Platform Chosen & Rationale
- **Provider:** Google Colab (`google-colab-cli` via `.\.venv\Scripts\colab.exe`, Pro+ compute-unit pool `~1,479 CU`).
- **Hardware:** `1× NVIDIA L4` (`24 GB` VRAM, Ada Lovelace `cc=8.9`, **native BF16**, `torch 2.14.1+cu130` pre-installed).
- **Session Name:** `acad-b-gauge-rope`
- **Estimated Wall-Clock & Cost:** ~25–35 minutes total (~10–12 min per 1,000-step arm + ~6–10 min for Rust build + `val` + warped-val + 45-piece `suite31_hw` eval) ≈ `~2.5–3.0 CU` (~0.2% of pool).
- **Rationale:** Native BF16, zero local GPU usage (`$env:CUDA_VISIBLE_DEVICES=""`), fast ~8s cold start with `torch` pre-installed, and immediate teardown via `colab stop -s acad-b-gauge-rope` in `finally`.

## 2. Identifiers for Follow-Up Queries
- **Colab Session:** `acad-b-gauge-rope` (stopped cleanly on exit; fell back to Modal `2× L4` due to 2-session L4 cap + Colab A100/H100 503s)
- **Modal App ID & URL:** `ap-woqJUva6rjQCbQnyydluQ5` (`https://modal.com/apps/gregoreesmaa1/main/ap-woqJUva6rjQCbQnyydluQ5`, `run_modal.py`, completed cleanly)
- **Hugging Face Repo:** `gregoreesmaa/sheet-music-converter-omr` (`repo_type="dataset"`)
  - Full Eval JSON: `evals/2026-11-26T01-30-academic-gauge-rope.json`
  - Run Ledger JSON: `runs/2026-11-26T01-30-academic-gauge-rope.json`
- **Remote Paths on Colab VM:**
  - Code bundle: `/content/acad_b_bundle.tar.gz` → extracted to `/workspace/`
  - Datasets & rolling ckpts: `/content/vol/`
  - Transient token file: `/tmp/.hf_token` (`chmod 600`, purged immediately on read and in `finally`)

### Copy-Paste Monitoring & Control Commands (PowerShell)
```powershell
# Check active Colab sessions & compute-unit balance
.\.venv\Scripts\colab.exe sessions
.\.venv\Scripts\colab.exe usage

# Stop Colab session immediately (ALWAYS run in finally!)
.\.venv\Scripts\colab.exe stop -s acad-b-gauge-rope

# Fetch completed eval JSON from Hugging Face
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; p = hf_hub_download('gregoreesmaa/sheet-music-converter-omr', 'evals/2026-11-26T01-30-academic-gauge-rope.json', repo_type='dataset'); print(open(p).read())"
```

## 3. Required Training & Eval Data
- **Supervised Proxy Dataset:** `datasets/iter10_5_proxy_2k.tar` (`2,000` train + `64` val full pages, permissive only, 0 `suite31_hw` overlap).
- **Unlabeled Real HW Dataset:** `datasets/rism_hw_5707.tar` (`5,707` public-domain historical RISM manuscript pages for `diffeomorphic_equivariance_loss`).
- **OOD Evaluation Benchmark:** `datasets/suite31_hw_eval.tar` (45 pieces: 31 printed + 14 real historical handwritten manuscripts, `clean` + `photo` = 90 pages).
- **Initialization:** Scratch `UniOMRv2Net(UniOMRv2Config.small_45m())` with identical `seed=42` for both arms (zero warm-start bias).

## 4. Exact Launch Command & Hyperparameters
```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\colab.exe new -s acad-b-gauge-rope --gpu L4
.\.venv\Scripts\colab.exe upload -s acad-b-gauge-rope <local_bundle.tar.gz> /content/acad_b_bundle.tar.gz
.\.venv\Scripts\colab.exe upload -s acad-b-gauge-rope <local_tmp_token> /tmp/.hf_token
.\.venv\Scripts\colab.exe exec -s acad-b-gauge-rope -f experiments/2026-11-26T01-30-academic-gauge-rope/run_colab.py
.\.venv\Scripts\colab.exe stop -s acad-b-gauge-rope
```

| Hyperparameter | Arm A (`control_rope2d`) | Arm B (`gauge_rope_rism_ssl`) |
|---|---|---|
| Architecture | `UniOMR v2 small_45m` (`d_model=384`, 2 axial + 4 global + 8 GQA dec, MTP-4) | Identical + `GaugeFieldHead` (`0.139M` params, `α_init=0.0`) + `GaugeRoPE2D` |
| Positional Encoding | Standard Euclidean `AspectNormalizedRoPE2D(y, x)` | Continuous `GaugeRoPE2D(y + α·Δv̂, x + α·Δû)` |
| Unlabeled RISM SSL | Off | `diffeomorphic_equivariance_loss` every 2 steps (`weight=0.25`, `B=2`, `max_dim=768`) |
| Steps | `1,000` | `1,000` |
| Effective Batch Size | `8` (`batch_size=2` × `grad_accum=4`) | `8` (`batch_size=2` × `grad_accum=4`) |
| Resolution & Seq Len | `fullpage_max_dim=1280`, `max_seq_len=1536`, `grad_checkpoint=True` | Identical |
| Precision | Native `BF16` (with FP32 trig in `GaugeRoPE2D`) | Identical |
| Optimizer & Schedule | Fused `AdamW` (`lr=3e-4`, `wd=0.04`), WSD (`warmup=100`, `decay=25%`, `min_lr=0.05`) | Identical (`wd=0.0` on `alpha`) |
| Seed & Batch Order | `seed=42` (identical init & supervised batch stream) | Identical |

## 5. 5-Minute Checkpointing & Resume Plan
- Every `5.0` minutes during each arm, `run_colab.py` atomically writes `uni_omr_v2_latest.pt` + `uni_omr_v2_ema_latest.pt` (`.tmp` → `os.replace`) to `/content/vol/ckpts/<arm>/`.
- Completed arm results are cached to `/content/vol/evals/<arm>_result.json` so re-executing `run_colab.py` in the same session skips completed arms cleanly.

## 6. Evaluation Gate & Decision Tree
- **Primary Metrics:**
  1. `proxy_val` (64 pages): `val_loss`, `val_ce_loss`, `val_token_f1`, `val_ser`
  2. **Warped-Val Robustness (3-staff-space droop, `droop_frac=0.025`):** `warped_val_ce_loss`, `warped_val_token_f1`, and droop degradation delta (`warped - clean`)
  3. **`suite31_hw` (45 pieces × 2 domains = 90 pages):** `printed_clean`, `printed_photo`, `hw_clean`, `hw_photo` — reporting **Strict Note F1**, **Token F1**, **Pitch-only Token F1**, **Rhythm-only Token F1**, and learned gate `|α|`.
- **Decision Tree:**
  - **PROMOTE:** Arm B beats Arm A on warped-val droop robustness (`droop_ce_delta` smaller or `warped_val_token_f1` higher) AND beats or matches Arm A on `suite31_hw` HW Pitch-only F1 / Strict Note F1 without regressing printed F1 → integrate `Gauge-RoPE` + RISM diffeomorphic SSL into Iter-11 `UniOMR v2`.
  - **ITERATE:** `|α|` moves cleanly and warped-val robustness improves, but 1,000-step scratch budget is too short for full `suite31_hw` Note F1 separation → test warm-start from `UniOMR v2` supervised checkpoint + longer RISM SSL schedule.
  - **DISCARD:** `|α|` stays near 0 or Arm B degrades both warped-val and `suite31_hw` relative to Arm A → document root cause in `OUTCOME.md` and keep Euclidean `AspectNormalizedRoPE2D`.
