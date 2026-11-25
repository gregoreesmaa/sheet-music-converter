# Experiment Plan — `2026-11-26T01-38-ov-29-ink-fade-bleedthrough-aug-prob`

> **Zero-Prior-Context Handoff:** Everything required to run, monitor, stop, resume, or evaluate Experiment 29 (Yellowed-Paper + Ink-Fade + Mirrored Bleed-Through `aug_prob` Sweep) is self-contained in this folder (`README.md`, `PLAN.md`, `run_colab.py`).

## 1. Platform Chosen & Rationale
- **Provider:** Google Colab (`google-colab-cli` via `.\.venv\Scripts\colab.exe`).
- **Hardware:** `1× NVIDIA L4` (`24 GB` VRAM, Ada Lovelace `cc=8.9`, **native BF16**, `torch 2.14.1+cu130` pre-installed).
- **Session Name:** `ov-29-fade-bleed`
- **Estimated Wall-Clock & Cost:** ~32–38 minutes total (~6.5–7.5 min per 600-step arm × 3 arms + ~9–12 min for Rust build + `proxy_val` clean/stress + 45-piece `suite31_hw` eval across 3 arms) ≈ `~2.6–3.1 CU`.
- **Rationale:** Native BF16, zero local GPU usage (`$env:CUDA_VISIBLE_DEVICES=""`), fast ~8s cold start, automatic teardown via `colab stop -s ov-29-fade-bleed`.

## 2. Identifiers for Follow-Up Queries
- **Colab Session:** `ov-29-fade-bleed`
- **Hugging Face Repo:** `gregoreesmaa/sheet-music-converter-omr` (`repo_type="dataset"`)
  - Full Eval JSON: `evals/2026-11-26T01-38-ov-29-ink-fade-bleedthrough-aug-prob.json`
  - Run Ledger JSON: `runs/2026-11-26T01-38-ov-29-ink-fade-bleedthrough-aug-prob.json`
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
.\.venv\Scripts\colab.exe stop -s ov-29-fade-bleed

# Fetch completed eval JSON from Hugging Face
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; p = hf_hub_download('gregoreesmaa/sheet-music-converter-omr', 'evals/2026-11-26T01-38-ov-29-ink-fade-bleedthrough-aug-prob.json', repo_type='dataset'); print(open(p).read())"
```

## 3. Required Training & Eval Data
- **Supervised Proxy Dataset:** `datasets/iter10_5_proxy_2k.tar` (`2,000` train + `64` val full pages, permissive only, 0 `suite31_hw` overlap).
- **OOD Evaluation Benchmark:** `datasets/suite31_hw_eval.tar` (45 pieces: 31 printed + 14 real historical handwritten manuscripts, `clean` + `photo` = 90 pages).
- **Initialization:** Scratch `UniOMRv2Net(UniOMRv2Config.small_45m(grad_checkpoint=True))` with identical `seed=42` across all 3 arms.

## 4. Exact Launch Command & Hyperparameters
```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\colab.exe new -s ov-29-fade-bleed --gpu L4
.\.venv\Scripts\colab.exe upload -s ov-29-fade-bleed <local_bundle.tar.gz> /content/repo_bundle.tar.gz
.\.venv\Scripts\colab.exe upload -s ov-29-fade-bleed <local_tmp_token> /tmp/.hf_token
.\.venv\Scripts\colab.exe exec -s ov-29-fade-bleed -f experiments/2026-11-26T01-38-ov-29-ink-fade-bleedthrough-aug-prob/run_colab.py
.\.venv\Scripts\colab.exe stop -s ov-29-fade-bleed
```

| Hyperparameter | Arm A (`aug_prob=0.30`) | Arm B (`aug_prob=0.55`) | Arm C (`aug_prob=0.75`) |
|---|---|---|---|
| Architecture | `UniOMR v2 small_45m` (`~43M` params) | Identical | Identical |
| `aug_prob` (Yellowed+Fade+Bleed) | `0.30` | `0.55` | `0.75` |
| Yellowed Parchment + Foxing | Warm `(212..244, 196..232, 162..208)` + `16×16` mottle | Identical | Identical |
| Spatially Varying Ink Fade | Iron-gall RGB + fade `∈ [0.22, 0.58]` | Identical | Identical |
| Mirrored Verso Bleed-Through | H-flipped shifted (`±4..14 px`) blurred verso, `alpha ∈ [0.08, 0.22]` | Identical | Identical |
| Steps / Eff. Batch | `600` steps, `B=2 × ga=4 = 8` | Identical | Identical |
| Resolution & Seq Len | `fullpage_max_dim=1280`, `max_seq_len=1536` | Identical | Identical |
| Precision / Opt | Native `BF16`, Fused `AdamW` (`lr=3e-4`, `wd=0.04`, WSD) | Identical | Identical |

## 5. 5-Minute Checkpointing & Resume Plan
- Every `5.0` minutes during each arm, `run_colab.py` atomically writes `uni_omr_v2_latest.pt` + `uni_omr_v2_ema_latest.pt` (`.tmp` → `os.replace`) to `/content/vol/ckpts/2026-11-26T01-38-ov-29-ink-fade-bleedthrough-aug-prob/<arm>/`.
- Completed arm results are cached to `<arm_dir>/arm_result.json` so re-running `run_colab.py` in the same session skips completed arms cleanly.

## 6. Evaluation Gate & Decision Tree
- **Primary Metrics:**
  1. `proxy_val` (64 pages): clean `val_ce_loss`, `val_token_f1`, `val_ser`
  2. **Archival Fade+Bleed Stress Test** (yellowed paper + 45% iron-gall ink fade + 16% shifted blurred mirrored verso bleed-through): `stress_ce_loss`, `stress_token_f1`, and degradation delta (`stress - clean`)
  3. **`suite31_hw` (45 pieces × 2 domains = 90 pages):** `printed_clean`, `printed_photo`, `hw_clean`, `hw_photo` — **Strict Note F1**, **Token F1**, **Pitch-only Token F1**, **Rhythm-only Token F1**.
- **Decision Tree:**
  - **PROMOTE:** Winning `aug_prob` achieves highest composite `0.5 * all45_clean + 0.5 * all45_photo` Token/Note F1 (and strongest `hw_clean`/`hw_photo` gain without >0.005 `printed_clean` drop) → set as Iter-11 default `--aug-prob` and adopt spatially varying fade + shifted blurred bleed-through in `uni_omr.py`.
  - **ITERATE:** `0.75` wins `hw_photo` while `0.30` wins `printed_clean` by `> 0.015` → adopt domain-conditioned `aug_prob` (`0.40` printed, `0.75` HW) in Iter-11.
  - **DISCARD:** All 3 arms are within noise (`< 0.005` delta across all splits) → keep current default `aug_prob=0.75`.
