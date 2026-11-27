# PLAN — `2026-11-27T14-50-i15-44-xf-verifier-kv-rewind-mbr`

## 1. Platform & Budget
- **Platform:** Google Colab `1× NVIDIA A100` (or Local `RTX 4080 SUPER` 16 GB, native BF16). Zero training — pure inference proxy.
- **Estimated Wall-Clock:** ~8–12 min total on A100 / 4080 SUPER (2 arms × 12 `suite31_hw` pages).
- **Baseline Commit SHA:** `6055c49`.
- **Self-Containment:** All KV-rewind + Structure-Conditional MBR decoding code lives strictly inside `run_colab.py`; `tmp-train/mvd/` is untouched.

## 2. Identifiers for Follow-Up Queries
- **Experiment ID:** `2026-11-27T14-50-i15-44-xf-verifier-kv-rewind-mbr`
- **Hugging Face (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-27T14-50-i15-44-xf-verifier-kv-rewind-mbr.json`
  - Run Ledger: `runs/2026-11-27T14-50-i15-44-xf-verifier-kv-rewind-mbr.json`
  - Partial Arm Sync: `checkpoints/2026-11-27T14-50-i15-44-xf-verifier-kv-rewind-mbr/{greedy_fsa,kv_rewind_mbr}_arm_result.json`

## 3. Data, Checkpoint & 2-Arm Protocol
- **Checkpoint:** Tries `checkpoints/iter14-v2a-final/uni_omr_v2_ema_latest.pt` on HF, falling back to `checkpoints/iter14-v2a-mid/uni_omr_v2_ema_latest.pt` (or `UNIOMR_CKPT` local override).
- **Eval Subset:** `datasets/suite31_hw_eval.tar` (`suite31_hw_manifest.json`), deterministic 12-page hard subset (`6` printed + `6` handwritten, `photo` variant; `max_dim=1408`, `max_len=2048`, `max_mem_tokens=4096`). Permissive licenses only; zero `CC-BY-NC`.
- **Windows RTX 4080 SUPER Safety:** Disables `torch.backends.cudnn.enabled` and forces CUTLASS `EFFICIENT_ATTENTION` on `win32`.
- **Arm 0 (`greedy_fsa`):** Standard KV-cached FSA-constrained greedy decoding (`grammar.PrefixState(strict=False)`).
- **Arm 1 (`kv_rewind_mbr`):** Inline bar-duration/voice-sync + peak-entropy verifier during KV-cached decoding:
  - Snapshots self-attention KV length and `PrefixState` at every bar boundary (`<bos>`, `<sys>`, `bar*`).
  - Triggers rewind when a closing bar has `meter_err > 1e-3` or `voice_err > 1e-3`, or peak intra-bar token entropy `> 1.6` nats (max `2` retries/bar, max `24` rewinds/page).
  - Rewinds self-attention KV cache in-place to the bar boundary (reusing encoder cross-KV), generates `K=4` candidates (1 peak-entropy top-1 masked greedy + 3 temperature-sampled `T=0.7, top_p=0.9` FSA continuations), and commits the candidate maximizing Structure-Conditional MBR score (conservation bonus + log-prob minus asymmetric edit cost).

## 4. Verification & Launch Commands
CPU mock 1-page rewind smoke test (zero GPU):
```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe experiments\2026-11-27T14-50-i15-44-xf-verifier-kv-rewind-mbr\run_colab.py --smoke-test
```
Colab A100 / Local 4080 SUPER run:
```bash
python3 -u experiments/2026-11-27T14-50-i15-44-xf-verifier-kv-rewind-mbr/run_colab.py
```

## 5. Checkpointing & Remote HF Auto-Resume Plan
- Per-page incremental cache saved atomically (`*.tmp` → `os.replace`) under `WORK_ROOT/ckpts/<arm>/pages.json`.
- **1-Arm HF Partial Sync & Startup Auto-Resume:** On startup, checks HF Private Hub for `evals/<EXP_ID>.json` and `checkpoints/<EXP_ID>/{greedy_fsa,kv_rewind_mbr}_arm_result.json` and skips completed arms. Uploads Arm 0 (`greedy_fsa`) results and partial `evals/<EXP_ID>.json` (`status: "RUNNING"`) to HF immediately before starting Arm 1.
- Security: reads `HF_TOKEN` from `/tmp/.hf_token` (chmod 600), `HF_TOKEN` env, or `~/.cache/huggingface/token`, never prints it, always unlinks `/tmp/.hf_token` in `finally`.

## 6. Evaluation Gate & Decision Tree
- **Primary Metrics:** `Note F1` (exact pitch+duration), `TokLCS F1`, `BSVR` (Bar-Sum Violation Rate, lower is better), `wFPP` (weighted Fixes-Per-Page proxy, lower is better), and wall-clock `ms/page`.
- **PROMOTE:** `kv_rewind_mbr` improves `Note F1` by `>= +0.010` OR cuts `BSVR` by `>= -25%` relative and improves `wFPP` with `<= 1.8×` latency → wire inline KV-rewind MBR into `eval_uni_suite31.py` / `eval_northstar_runner.py` and production decoder.
- **ITERATE:** Cuts `BSVR` cleanly (`>= -15%` relative) but latency `> 1.8×` or neutral `Note F1` → tune entropy threshold / `K=2` / violation-only trigger (skip entropy-only rewinds).
- **DISCARD:** `< +0.003` `Note F1` and `< -10%` `BSVR` reduction, or `wFPP` regression → keep standard FSA greedy / TTA reranking.
