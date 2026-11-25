# Experiment Plan — `2026-11-26T01-38-ov-21-paper-gate-border-threshold`

> **Zero-Prior-Context Handoff:** This folder (`README.md`, `PLAN.md`, `run_colab.py`) is 100% self-contained. Any agent can launch, monitor, stop, or inspect Overnight Exp 21 without prior conversation context.

## 1. Platform Chosen & Rationale
- **Provider:** Google Colab (`.\.venv\Scripts\colab.exe`, Pro+ compute-unit pool).
- **Hardware:** `1× NVIDIA L4` (`24 GB` VRAM, Ada Lovelace `cc=8.9`, `torch 2.14.1+cu130` pre-installed).
- **Session Name:** `ov-21-paper-gate-sweep`
- **Estimated Wall-Clock & Cost:** ~15–25 minutes total (download/unpack `suite31_hw_eval.tar` + `rism_hw_5707.tar`, resize 290 pages to `1280px`, run 9×290 = 2,610 GPU flood-fill passes + parity checks + HF upload) ≈ `~1.5–2.0 CU`.
- **Rationale:** Zero training needed; GPU-accelerated pure-float32 tensor ops evaluate all 9 threshold pairs across 290 real/OOD full pages deterministically with zero local GPU usage (`$env:CUDA_VISIBLE_DEVICES=""`).

## 2. Identifiers for Follow-Up Queries
- **Colab Session:** `ov-21-paper-gate-sweep`
- **Hugging Face Repo:** `gregoreesmaa/sheet-music-converter-omr` (`repo_type="dataset"`)
  - Full Eval JSON: `evals/2026-11-26T01-38-ov-21-paper-gate-border-threshold.json`
  - Run Ledger JSON: `runs/2026-11-26T01-38-ov-21-paper-gate-border-threshold.json`
- **Remote Paths on Colab VM:**
  - Code bundle (optional): `/content/bundle.tar.gz` → `/workspace/`
  - Datasets & outputs: `/content/vol/`
  - Transient token file: `/tmp/.hf_token` (`chmod 600`, deleted immediately on read and in `finally`)

### Copy-Paste Monitoring & Control Commands (PowerShell)
```powershell
# Check active Colab sessions & compute-unit balance
.\.venv\Scripts\colab.exe sessions
.\.venv\Scripts\colab.exe usage

# Stop Colab session immediately (ALWAYS run in finally!)
.\.venv\Scripts\colab.exe stop -s ov-21-paper-gate-sweep

# Fetch completed eval JSON from Hugging Face
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; p = hf_hub_download('gregoreesmaa/sheet-music-converter-omr', 'evals/2026-11-26T01-38-ov-21-paper-gate-border-threshold.json', repo_type='dataset'); print(open(p).read())"
```

## 3. Required Eval Data
- **OOD Evaluation Benchmark:** `gregoreesmaa/sheet-music-converter-omr` → `datasets/suite31_hw_eval.tar` (`45` pieces × `clean` + `photo` = `90` pages: 31 printed + 14 real historical HW manuscripts, 100% permissive).
- **Real Historical Manuscripts:** `gregoreesmaa/sheet-music-converter-omr` → `datasets/rism_hw_5707.tar` (`200` pages deterministically sampled with `seed=42` from the sorted list of 5,707 public-domain RISM pages).

## 4. Exact Launch Command & Sweep Grid
```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\colab.exe new -s ov-21-paper-gate-sweep --gpu L4
.\.venv\Scripts\colab.exe upload -s ov-21-paper-gate-sweep <local_bundle.tar.gz> /content/bundle.tar.gz
.\.venv\Scripts\colab.exe upload -s ov-21-paper-gate-sweep <local_tmp_token> /tmp/.hf_token
.\.venv\Scripts\colab.exe exec -s ov-21-paper-gate-sweep -f experiments/2026-11-26T01-38-ov-21-paper-gate-border-threshold/run_colab.py
.\.venv\Scripts\colab.exe stop -s ov-21-paper-gate-sweep
```

| Parameter | Values Swept | Current Default |
|---|---|---:|
| `luma_stop` (edge-scan bright stop threshold) | `0.35`, `0.45`, `0.55` | `0.45` |
| `dark_thr` (dark exterior pixel threshold) | `0.22`, `0.30`, `0.38` | `0.30` |
| `bg_floor` | `0.05` (fixed) | `0.05` |
| Image resolution (`fullpage_max_dim`) | `1280` (padded to `/16` grid) | `1280` |
| Total grid points × pages | `9 × 290 = 2,610` evaluations | — |

## 5. Checkpointing & Resume Plan
- Each of the 9 `(luma_stop, dark_thr)` grid points writes an atomic per-config cache JSON to `/content/vol/evals/2026-11-26T01-38-ov-21-paper-gate-border-threshold/cfg_<luma>_<dark>.json` immediately upon completion (`.tmp` → `os.replace`), so re-running `run_colab.py` skips completed grid points.

## 6. Evaluation Gate & Decision Tree
- **Primary Metrics (per cohort & overall):**
  1. `interior_ink_recall_mean` & `interior_ink_recall_p05` (target: `>= 0.995` on `suite31_hw`, `>= 0.990` on RISM)
  2. `desk_border_suppression_mean` on `suite31_hw` `photo` & RISM exterior dark borders (target: `>= 0.95`)
  3. `clean_false_mask_rate` on `suite31_hw` `clean` pages (target: `<= 0.005`)
  4. `collapsed_page_count` (`effective_paper_f.mean() < 0.50`, target: `0` out of 290 pages)
  5. `top40_ink_patch_recall` (simulated router top-40% ink recall)
- **Decision Tree:**
  - **PROMOTE:** If a non-default `(luma_stop, dark_thr)` strictly improves RISM + `hw_photo` `interior_ink_recall_p05` or eliminates collapsed pages while keeping `desk_border_suppression >= 0.95` and `clean_false_mask_rate <= 0.005`, update both `vir_router_v2.py` and `uni_omr_v2.py` (plus unit tests) before Iter-11 training.
  - **ITERATE:** If single global thresholds still struggle on extreme dark-parchment RISM scans (`collapsed_page_count > 0` across all 9 configs), design a percentile-relative adaptive `luma_stop` (`0.6 * p85(luma)`).
  - **DISCARD:** If `(0.45, 0.30)` already sits on the strict Pareto frontier with `0` collapsed pages and `>= 0.995` interior ink recall, retain current defaults and document proof in `OUTCOME.md`.
