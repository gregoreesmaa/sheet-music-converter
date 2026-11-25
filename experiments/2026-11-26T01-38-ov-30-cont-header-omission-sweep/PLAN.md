# Experiment Plan — `2026-11-26T01-38-ov-30-cont-header-omission-sweep`

> **Zero-Prior-Context Handoff:** Everything required to run, monitor, stop, resume, or evaluate Experiment 30 (Continuation-Header Omission Sweep on Systems 2+) is self-contained in this folder (`README.md`, `PLAN.md`, `run_colab.py`).

## 1. Platform Chosen & Rationale
- **Provider:** Google Colab (`google-colab-cli` via `.\.venv\Scripts\colab.exe`).
- **Hardware:** `1× NVIDIA L4` (`24 GB` VRAM, Ada Lovelace `cc=8.9`, **native BF16**, `torch 2.14.1+cu130` pre-installed).
- **Session Name:** `ov-30-cont-header`
- **Estimated Wall-Clock & Cost:** ~32–38 minutes total (~6.5–7.5 min per 600-step arm × 3 arms + ~9–12 min for Rust build + `proxy_val` clean/omitted-header stress + 45-piece `suite31_hw` eval across 3 arms) ≈ `~2.6–3.1 CU`.
- **Rationale:** Native BF16, zero local GPU usage (`$env:CUDA_VISIBLE_DEVICES=""`), fast ~8s cold start, automatic teardown via `colab stop -s ov-30-cont-header`.

## 2. Identifiers for Follow-Up Queries
- **Colab Session:** `ov-30-cont-header`
- **Hugging Face Repo:** `gregoreesmaa/sheet-music-converter-omr` (`repo_type="dataset"`)
  - Full Eval JSON: `evals/2026-11-26T01-38-ov-30-cont-header-omission-sweep.json`
  - Run Ledger JSON: `runs/2026-11-26T01-38-ov-30-cont-header-omission-sweep.json`
- **Remote Paths on Colab VM:**
  - Code bundle (optional): `/content/repo_bundle.tar.gz` → extracted to `/workspace/`
  - Datasets & rolling ckpts: `/content/vol/`
  - Transient token file: `/tmp/.hf_token` (`chmod 600`, purged immediately on read and in `finally`)

### Copy-Paste Monitoring & Control Commands (PowerShell)
```powershell
# Check active Colab sessions & compute-unit balance
.\.venv\Scripts\colab.exe sessions
.\.venv\Scripts\colab.exe usage

# Stop Colab session immediately (ALWAYS run in finally!)
.\.venv\Scripts\colab.exe stop -s ov-30-cont-header

# Fetch completed eval JSON from Hugging Face
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; p = hf_hub_download('gregoreesmaa/sheet-music-converter-omr', 'evals/2026-11-26T01-38-ov-30-cont-header-omission-sweep.json', repo_type='dataset'); print(open(p).read())"
```

## 3. Required Training & Eval Data
- **Supervised Proxy Dataset:** `datasets/iter10_5_proxy_2k.tar` (`2,000` train + `64` val full pages, permissive only, 0 `suite31_hw` overlap).
- **OOD Evaluation Benchmark:** `datasets/suite31_hw_eval.tar` (45 pieces: 31 printed + 14 real historical handwritten manuscripts, `clean` + `photo` = 90 pages).
- **Initialization:** Scratch `UniOMRv2Net(UniOMRv2Config.small_45m(grad_checkpoint=True))` with identical `seed=42` across all 3 arms.

## 4. Exact Launch Command & Hyperparameters
```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\colab.exe new -s ov-30-cont-header --gpu L4
.\.venv\Scripts\colab.exe upload -s ov-30-cont-header <local_bundle.tar.gz> /content/repo_bundle.tar.gz
.\.venv\Scripts\colab.exe upload -s ov-30-cont-header <local_tmp_token> /tmp/.hf_token
.\.venv\Scripts\colab.exe exec -s ov-30-cont-header -f experiments/2026-11-26T01-38-ov-30-cont-header-omission-sweep/run_colab.py
.\.venv\Scripts\colab.exe stop -s ov-30-cont-header
```

| Hyperparameter | Arm A (`omit=0.00`) | Arm B (`omit=0.35`) | Arm C (`omit=0.60`) |
|---|---|---|---|
| Architecture | `UniOMR v2 small_45m` (`~43M` params) | Identical | Identical |
| `omit_cont_header_prob` (Systems 2+) | `0.00` (Control) | `0.35` | `0.60` |
| Header Erase Region | None | Left `5–9%` of active system width, local paper fill (System 1 untouched) | Identical |
| Ground-Truth Targets | Full canonical bekern (carry-forward required) | Identical | Identical |
| Base Online Aug | `aug_prob=0.55` | `aug_prob=0.55` | `aug_prob=0.55` |
| Steps / Eff. Batch | `600` steps, `B=2 × ga=4 = 8` | Identical | Identical |
| Resolution & Seq Len | `fullpage_max_dim=1280`, `max_seq_len=1536` | Identical | Identical |
| Precision / Opt | Native `BF16`, Fused `AdamW` (`lr=3e-4`, `wd=0.04`, WSD) | Identical | Identical |

## 5. 5-Minute Checkpointing & Resume Plan
- Every `5.0` minutes during each arm, `run_colab.py` atomically writes `uni_omr_v2_latest.pt` + `uni_omr_v2_ema_latest.pt` (`.tmp` → `os.replace`) to `/content/vol/ckpts/2026-11-26T01-38-ov-30-cont-header-omission-sweep/<arm>/`.
- Completed arm results are cached to `<arm_dir>/arm_result.json` so re-running `run_colab.py` in the same session skips completed arms cleanly.

## 6. Evaluation Gate & Decision Tree
- **Primary Metrics:**
  1. `proxy_val` (64 pages): clean `val_ce_loss`, `val_token_f1`, `val_cont_header_f1` (Clef+Key F1 on Systems 2+)
  2. **Continuation-Header-Omitted Stress Test** (all Systems 2+ have left `7%` header erased on `proxy_val`): `stress_ce_loss`, `stress_token_f1`, `stress_pitch_token_f1`, `stress_cont_header_f1`, and degradation delta (`stress - clean`)
  3. **`suite31_hw` (45 pieces × 2 domains = 90 pages):** `printed_clean`, `printed_photo`, `hw_clean`, `hw_photo` — **Strict Note F1**, **Token F1**, **Pitch-only Token F1**, **Rhythm-only Token F1**.
- **Decision Tree:**
  - **PROMOTE:** Winning non-zero arm (`0.35` or `0.60`) improves `stress_cont_header_f1` by `≥ +0.05` AND improves `suite31_hw` `hw_clean` or `hw_photo` Pitch/Note F1 with `≤ 0.005` regression on `printed_clean` → wire `omit_cont_header_prob` into `UniFullPageDataset` for Iter-11.
  - **ITERATE:** `stress_cont_header_f1` improves strongly, but `0.60` hurts printed scores with mid-page clef changes → restrict header omission to continuation systems whose clef/key matches System 1 (verified from bekern tokens) at `p=0.35`.
  - **DISCARD:** Neither `0.35` nor `0.60` improves `suite31_hw` HW Pitch/Note F1 or continuation-header carry-forward → drop visual header erasure.
