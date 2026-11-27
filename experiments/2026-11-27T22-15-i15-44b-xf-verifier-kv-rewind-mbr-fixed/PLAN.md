# PLAN — `2026-11-27T22-15-i15-44b-xf-verifier-kv-rewind-mbr-fixed`

## 1. Platform & Budget
- **Platform:** Local `RTX 4080 SUPER` 16 GB (Windows 11, native BF16, `--no-cudnn` + CUTLASS `EFFICIENT_ATTENTION`) or Google Colab `1× NVIDIA A100`. Zero training — pure inference proxy.
- **Estimated Wall-Clock:** ~5–8 min total on RTX 4080 SUPER / A100 (2 arms × 12 `suite31_hw` pages, targeting `<= 1.35×` overhead on Arm 1).
- **Baseline Commit SHA:** `ca8826e`.
- **Self-Containment:** All KV-rewind + Structure-Conditional MBR decoding and MusicXML scoring code lives strictly inside `run_colab.py`; `tmp-train/mvd/` is untouched.

## 2. Identifiers for Follow-Up Queries
- **Experiment ID:** `2026-11-27T22-15-i15-44b-xf-verifier-kv-rewind-mbr-fixed`
- **Hugging Face (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-27T22-15-i15-44b-xf-verifier-kv-rewind-mbr-fixed.json`
  - Run Ledger: `runs/2026-11-27T22-15-i15-44b-xf-verifier-kv-rewind-mbr-fixed.json`
  - Partial Arm Sync: `checkpoints/2026-11-27T22-15-i15-44b-xf-verifier-kv-rewind-mbr-fixed/{greedy_fsa,kv_rewind_mbr}_arm_result.json`

## 3. Data, Checkpoint & 2-Arm Protocol
- **Checkpoint:** Tries `checkpoints/iter14-v2a-final/uni_omr_v2_ema_latest.pt` on HF, falling back to `checkpoints/iter14-v2a-mid/uni_omr_v2_ema_latest.pt` (or `UNIOMR_CKPT` local override).
- **Eval Subset:** `datasets/suite31_hw_eval.tar` (`suite31_hw_manifest.json`), deterministic 12-page hard subset (`6` printed + `6` handwritten, `photo` variant; `max_dim=1408`, `max_len=2048`, `max_mem_tokens=4096`). Permissive licenses only; zero `CC-BY-NC`.
- **Scoring (Bug #1 Fix):**
  - Uses `eval_uni_suite31` + `eval_sys_gate` (`suite31_page_refs` → `assembly-serialize bekern-decode` → `eval-harness strict-f1` / `score_notes_pair`) when compiled Rust binaries are present, reporting true Strict `note_f1`, `tok_lcs_f1`, `bsvr`, and `wfpp`.
  - Falls back cleanly to pure-Python `xml.etree.ElementTree` MusicXML note/header extraction + `rlvr_grpo.extract_pitch_dur_notes` + `rlvr_grpo.compute_sequence_f1` on Colab Linux (never `piece["gt"].read_text().split()`).
- **Windows RTX 4080 SUPER Safety:** Disables `torch.backends.cudnn.enabled` and forces CUTLASS `EFFICIENT_ATTENTION` on `win32`.
- **Arm 0 (`greedy_fsa`):** Standard KV-cached FSA-constrained greedy decoding (`grammar.PrefixState(strict=False)`).
- **Arm 1 (`kv_rewind_mbr`, Bug #2 Fix):** Violation-only inline KV-cache rewind + Structure-Conditional MBR:
  - Snapshots self-attention KV length and `PrefixState` at every bar boundary (`<bos>`, `<sys>`, `bar*`).
  - Triggers rewind **ONLY** when a closing bar has an actual conservation violation (`meter_err > 1e-3` or `voice_err > 1e-3`); removes the noisy `peak_ent > 1.60` trigger.
  - Caps at `K=2` candidates (1 peak-entropy top-1 masked greedy + 1 temperature-sampled `T=0.70, top_p=0.90` FSA continuation), `MAX_RETRIES_PER_BAR=1`, `MAX_REWINDS_PER_PAGE=8`.
  - Hard-bans `<eos>` inside `_rollout_bar_candidate` (`ban_eos=True`) and applies `-100.0` penalty to any premature-`<eos>` candidate (`-50.0` for unclosed rollouts) in `_select_mbr_bar_candidate` so premature `<eos>` can never truncate a page.

## 4. Verification & Launch Commands
CPU mock MusicXML + premature-`<eos>` rejection + rewind smoke test (zero GPU):
```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe experiments\2026-11-27T22-15-i15-44b-xf-verifier-kv-rewind-mbr-fixed\run_colab.py --smoke-test
```
Local RTX 4080 SUPER / Colab A100 run:
```bash
python3 -u experiments/2026-11-27T22-15-i15-44b-xf-verifier-kv-rewind-mbr-fixed/run_colab.py
```

## 5. Checkpointing & Remote HF Auto-Resume Plan
- Per-page incremental cache saved atomically (`*.tmp` → `os.replace`) under `WORK_ROOT/ckpts/<arm>/pages.json`.
- **1-Arm HF Partial Sync & Startup Auto-Resume:** On startup, checks HF Private Hub for `evals/<EXP_ID>.json` and `checkpoints/<EXP_ID>/{greedy_fsa,kv_rewind_mbr}_arm_result.json` and skips completed arms. Uploads Arm 0 (`greedy_fsa`) results and partial `evals/<EXP_ID>.json` (`status: "RUNNING"`) to HF immediately before starting Arm 1.
- Security: reads `HF_TOKEN` from `/tmp/.hf_token` (chmod 600), `HF_TOKEN` env, or `~/.cache/huggingface/token`, never prints it, always unlinks `/tmp/.hf_token` in `finally`.

## 6. Evaluation Gate & Decision Tree
- **Primary Metrics:** `Note F1` (real MusicXML note-matching), `TokLCS F1`, `BSVR` (Bar-Sum Violation Rate, lower is better), `wFPP` (weighted Fixes-Per-Page proxy, lower is better), and wall-clock `ms/page` (`latency_ratio`).
- **PROMOTE:** `kv_rewind_mbr` improves overall `Note F1` by `>= +0.008` (with no printed `Note F1` regression `> -0.003`) OR cuts overall `BSVR` by `>= -25%` relative with non-negative `Note F1` and improved `wFPP` at `<= 1.35×` latency → wire inline violation-only KV-rewind MBR into `eval_uni_suite31.py` / production decoder.
- **ITERATE:** Cuts `BSVR` cleanly (`>= -15%` relative) with non-negative `Note F1` but latency `> 1.35×` or mixed printed/HW trade-off → gate rewind strictly by domain/bar length or integrate with `MeasureBeamHeaderGuard`.
- **DISCARD:** `< +0.003` `Note F1` and `< -10%` `BSVR` reduction, or `Note F1` / `wFPP` regression → keep standard FSA greedy / promoted `mbeam2` decoder.
