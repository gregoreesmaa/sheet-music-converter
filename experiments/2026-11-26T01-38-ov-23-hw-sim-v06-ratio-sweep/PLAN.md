# Experiment Plan — `2026-11-26T01-38-ov-23-hw-sim-v06-ratio-sweep`

> **Zero-Prior-Context Handoff:** Everything required to launch, monitor, resume, stop, or evaluate Overnight Experiment 23 is self-contained in this folder (`README.md`, `PLAN.md`, `run_colab.py`).

## 1. Platform Chosen & Rationale
- **Provider:** Google Colab (`google-colab-cli` via `.\.venv\Scripts\colab.exe`, Pro+ compute-unit pool).
- **Hardware:** `1x NVIDIA L4` (`24 GB` VRAM, Ada Lovelace `cc=8.9`, **native BF16**, `torch 2.14.1+cu130` pre-installed).
- **Session Name:** `ov-23-hw-v06-ratio`
- **Estimated Wall-Clock & Cost:** `~34-40 min` total (`~9-10 min` per `800`-step arm x 3 arms + `~6-8 min` for `proxy_val` + `hw_sim_v06_val` + 90-page `suite31_hw` zero-shot eval) ≈ `~2.8-3.3 CU`.
- **Rationale:** Native BF16 + 24 GB VRAM handles `UniOMR v2 small_45m` + full `suite31_hw` zero-shot inference cleanly without touching the local RTX 4080 SUPER.

## 2. Identifiers for Follow-Up Queries
- **Colab Session:** `ov-23-hw-v06-ratio`
- **Hugging Face Repo (`dataset`):** `gregoreesmaa/sheet-music-converter-omr`
  - Eval JSON: `evals/2026-11-26T01-38-ov-23-hw-sim-v06-ratio-sweep.json`
  - Run Ledger JSON: `runs/2026-11-26T01-38-ov-23-hw-sim-v06-ratio-sweep.json`
- **Remote Paths on Colab VM:**
  - Code bundle: `/content/repo_bundle.tar.gz` -> extracted to `/workspace/sheet-music-converter`
  - Datasets: `/content/vol/datasets/{iter10_5_proxy_2k,hw_sim_v06_500,suite31_hw_eval}/`
  - Mixed manifests & checkpoints: `/content/vol/ckpts/2026-11-26T01-38-ov-23-hw-sim-v06-ratio-sweep/<arm>/`
  - Transient token file: `/tmp/.hf_token` (`chmod 600`, deleted immediately after read AND guaranteed deleted in `finally`)

### Copy-Paste Monitoring & Retrieval Commands (PowerShell)
```powershell
# Check active Colab sessions & compute-unit balance
.\.venv\Scripts\colab.exe sessions
.\.venv\Scripts\colab.exe usage

# Stop Colab session immediately (ALWAYS run in finally!)
.\.venv\Scripts\colab.exe stop -s ov-23-hw-v06-ratio

# Fetch completed eval JSON from Hugging Face
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; p = hf_hub_download('gregoreesmaa/sheet-music-converter-omr', 'evals/2026-11-26T01-38-ov-23-hw-sim-v06-ratio-sweep.json', repo_type='dataset'); print(open(p, encoding='utf-8').read())"
```

## 3. Required Training & Eval Data (100% Permissive, 0 Eval Overlap)
- **Printed Base Pool:** `datasets/iter10_5_proxy_2k.tar` (`proxy_train.jsonl` printed subset + `proxy_val.jsonl` 64-page held-out val split).
- **Historical-Clean HW-Sim v0.6:** `datasets/hw_sim_v06_500.tar` (deterministic split: `40` held-out `hw_sim_v06_val.jsonl` pages + remaining `~460` training pages, 100% permissive, barline-clean).
- **OOD Evaluation Benchmark:** `datasets/suite31_hw_eval.tar` (45 pieces: 31 printed + 14 real historical handwritten manuscripts, `clean` + `photo` = 90 pages).
- **Exact Ratio Enforcement:** `args.stratified_fullpage` is explicitly set to `False` so `train_uni_v2.py` samples directly from each arm's `2,000`-row mixed manifest at the exact `10%` / `25%` / `40%` HW ratio (no hidden 50/50 domain reweighting).

## 4. Exact Launch Command & Hyperparameters (Commit `164852f`)
```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\colab.exe new -s ov-23-hw-v06-ratio --gpu L4
.\.venv\Scripts\colab.exe upload -s ov-23-hw-v06-ratio <local_bundle.tar.gz> /content/repo_bundle.tar.gz
.\.venv\Scripts\colab.exe upload -s ov-23-hw-v06-ratio <local_tmp_token> /tmp/.hf_token
.\.venv\Scripts\colab.exe exec -s ov-23-hw-v06-ratio -f experiments/2026-11-26T01-38-ov-23-hw-sim-v06-ratio-sweep/run_colab.py
.\.venv\Scripts\colab.exe stop -s ov-23-hw-v06-ratio
```

| Hyperparameter | Arm A (`hw_10`) | Arm B (`hw_25`) | Arm C (`hw_40`) |
|---|---|---|---|
| `hw_sim_v06` fraction *(isolated variable)* | **`10%`** (`200 / 2,000`) | **`25%`** (`500 / 2,000`) | **`40%`** (`800 / 2,000`) |
| Printed proxy fraction | `90%` (`1,800 / 2,000`) | `75%` (`1,500 / 2,000`) | `60%` (`1,200 / 2,000`) |
| `--preset` / `--optimizer` | `small_45m` / `muon` | `small_45m` / `muon` | `small_45m` / `muon` |
| `--curriculum` / `stratified_fullpage` | `fullpage` / `False` | `fullpage` / `False` | `fullpage` / `False` |
| `--steps` / `--seed` | `800` / `42` | `800` / `42` | `800` / `42` |
| `--batch-size` / `--grad-accum` | `2` / `4` | `2` / `4` | `2` / `4` |
| `--fullpage-max-dim` / `--max-seq-len` | `1280` / `1536` | `1280` / `1536` | `1280` / `1536` |
| `--lr-schedule` / `--lr` / `--warmup-steps` | `wsd` / `3e-4` / `80` | `wsd` / `3e-4` / `80` | `wsd` / `3e-4` / `80` |
| `--augment --aug-prob` | `0.50` | `0.50` | `0.50` |
| `--grad-checkpoint --channels-last --bf16` | Enabled | Enabled | Enabled |
| `--save-every-minutes` | `5.0` | `5.0` | `5.0` |

## 5. 5-Minute Checkpointing & Resume Plan
- `train_uni_v2.py` writes `uni_omr_v2_latest.pt` + `uni_omr_v2_ema_latest.pt` atomically every `5.0` minutes into `/content/vol/ckpts/2026-11-26T01-38-ov-23-hw-sim-v06-ratio-sweep/<arm>/`.
- Completed arms persist `arm_result.json`; re-running `run_colab.py` skips completed arms automatically.

## 6. Evaluation Gate & Decision Tree
- **Metrics Measured (EMA checkpoint):**
  1. `proxy_val.jsonl` (`printed` vs `hw` vs `all`): Val CE, Val MTP, greedy SER, Token F1, Sys-Count Exact.
  2. `hw_sim_v06_val.jsonl` (40 held-out pages): Val CE, greedy SER, Token F1.
  3. **`suite31_hw` zero-shot transfer (45 pieces x 2 = 90 pages):** `printed_clean`, `printed_photo`, `hw_clean`, `hw_photo` — Token F1, Pitch-only F1, Rhythm-only F1, SER.
- **Decision Tree:**
  - **PROMOTE:** Lock the `hw_sim_v06` ratio that maximizes `suite31_hw` `hw_clean` + `hw_photo` Token F1 while keeping `printed_clean` + `printed_photo` within `<= 0.015` Token F1 of the best printed arm into the Iter-11 full manifest builder.
  - **ITERATE:** If `hw_40` still climbs steeply on `suite31_hw` HW transfer with minimal printed loss, run a follow-up `40%` vs `55%` + CUT-RISM translated `hw_sim_v06` comparison.
  - **DISCARD:** If `hw_sim_v06` fails to beat `hw_10` on real `suite31_hw` HW transfer despite lower `hw_sim_v06_val` CE (indicating sim-to-real gap), prioritize `cut_rism_v06_500.tar` style-transferred pages instead of raw `hw_sim_v06`.
