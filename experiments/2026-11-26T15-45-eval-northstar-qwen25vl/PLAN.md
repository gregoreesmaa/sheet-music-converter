# PLAN — `2026-11-26T15-45-eval-northstar-qwen25vl`

## 1. Platform Chosen & Rationale

- **Platform:** **Modal L4** (`24 GB VRAM`, native BF16, `~$0.80/hr` within `$30/mo` free tier) via `experiments/overnight_queue/modal_runner.py` (also runs unmodified on Google Colab L4).
- **Model:** `Qwen/Qwen2.5-VL-7B-Instruct` (Apache-2.0, ungated; ~15.5 GB in BF16 with native PyTorch SDPA, `max_pixels=1280*28*28`, greedy decoding `do_sample=False`, `max_new_tokens=2048`).
  - *Why not `guangyangmusic/legato`:* Verified gated (`403 GatedRepoError` for `legato`, `legato-small`, and `legato-2` on `2026-11-26`).
- **Estimated Wall-Clock:** ~25–45 minutes for 125 full pages on L4 GPU (~12–20 s/page autoregressive generation + `music21` ABC → MusicXML + `omr_northstar` scoring).
- **Estimated Cost:** `~$0.35–$0.60` on Modal L4 free credit.

## 2. Identifiers for Follow-Up Queries

- **Experiment ID:** `2026-11-26T15-45-eval-northstar-qwen25vl`
- **HF Dataset Input:** `gregoreesmaa/sheet-music-converter-omr` → `datasets/northstar_eval_pack_125p.tar.gz`
- **HF Eval Output (canonical 4-layer report):** `evals/2026-11-26T15-45-eval-northstar-qwen25vl.json`
- **HF Per-Page Breakdown:** `evals/2026-11-26T15-45-eval-northstar-qwen25vl_per_page.json`
- **HF Run Ledger:** `runs/2026-11-26T15-45-eval-northstar-qwen25vl.json`

## 3. Exact Modal L4 Launch Commands

```powershell
# Detached Modal L4 launch (recommended — safe against local disconnects):
.\.venv\Scripts\modal.exe run --detach experiments/overnight_queue/modal_runner.py `
  --exp-id 2026-11-26T15-45-eval-northstar-qwen25vl `
  --script-path experiments/2026-11-26T15-45-eval-northstar-qwen25vl/run_colab.py `
  --hf-eval-target evals/2026-11-26T15-45-eval-northstar-qwen25vl.json `
  --gpu L4 `
  --timeout-secs 7200
```

```powershell
# Monitor running Modal app & logs:
.\.venv\Scripts\modal.exe app list
.\.venv\Scripts\modal.exe app logs <app-id> -f
```

## 4. Pipeline & Checkpointing Details

1. Downloads & unpacks `datasets/northstar_eval_pack_125p.tar.gz` to `/vol/datasets/northstar_eval_pack_125p/` (cached on Modal `omr-cache` Volume).
2. Ensures `transformers>=4.49.0`, `accelerate`, and `qwen-vl-utils` are installed without modifying PyTorch.
3. Loads `Qwen/Qwen2.5-VL-7B-Instruct` in `torch.bfloat16` (`attn_implementation="sdpa"`, `device_map="auto"`).
4. Prompts deterministically (`do_sample=False`, `max_new_tokens=2048`) to output either `NO_MUSIC` (if no sheet music is present) or valid ABC notation (`X:1`, `M:`, `L:`, `K:`, body).
5. Saves raw `<page_id>.abc`, converts ABC → MusicXML via `music21.converter.parseData(..., format="abc")` (`NO_MUSIC` → valid empty MusicXML), and writes `<page_id>.musicxml` atomically to `/vol/runs/2026-11-26T15-45-eval-northstar-qwen25vl/musicxml/` (resume-safe).
6. Scores all 125 pages using the bundled `omr_northstar` stdlib scorer and uploads the validated report + per-page breakdown + run ledger to HF Private Hub.

## 5. Evaluation Gate & Decision Tree

- **RECORD / BASELINE:** Record `Qwen/Qwen2.5-VL-7B-Instruct` in the OMR-NorthStar competitor leaderboard (`OUTCOME.md`, `progress.md`) as the open-weights general VLM baseline.
- **FAITHFUL FAILURE LOGGING:** Never hand-filter `music21` ABC parse errors or truncation; log exact `parseable=False` count as part of `CFR` and record conversion stats in `runs/<exp_id>.json`.
- **FOLLOW-UP:** Once HF access is granted for `guangyangmusic/legato`, clone this experiment for a direct Legato (specialized ABC VLM) comparison.
