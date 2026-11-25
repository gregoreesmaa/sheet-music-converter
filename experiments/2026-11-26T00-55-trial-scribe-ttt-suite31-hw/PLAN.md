# Experiment Plan — `2026-11-26T00-55-trial-scribe-ttt-suite31-hw`

## 1. Platform Chosen & Rationale
- **Platform:** **Modal `L4` (`24 GB`, `cc=8.9`, native BF16)** — `~$0.80/hr` within the `$30/mo` free tier.
- **Rationale:** Zero interference with the local RTX 4080 SUPER (which is running Iter 10.5), native BF16 support (unlike Kaggle T4), fast shared Modal Volume (`omr-cache`) for dataset/checkpoint caching, and ~10–15 min wall-clock (`~$0.15–$0.20` total credit).
- **Estimated Wall-Clock:** `~10–15 min` (28 pages × `[1 baseline decode + 3-step TTT + decode + 5-step TTT + decode]` on L4 BF16 with MTP-4 speculative decoding + first-time Linux Rust build of `assembly-serialize` and `eval-harness` cached in the image).

## 2. Identifiers for Follow-Up Queries (Zero-Prior-Context Handoff)
- **Modal Profile:** `gregoreesmaa1`
- **Modal App Name:** `exp6-trial-scribe-ttt-suite31-hw`
- **Modal App ID:** `ap-eLJrorKeWReglJ2HRrX7IB` (call ID: `fc-01M8AFT2HGFZQDE4F8ME2G19JR`, `ephemeral (detached)`, `NVIDIA L4`, `cc=8.9`, `bf16=True`)
- **Modal Dashboard URL:** `https://modal.com/apps/gregoreesmaa1/main/ap-eLJrorKeWReglJ2HRrX7IB`
- **Modal Shared Volume:** `omr-cache` (mounted at `/vol`)
- **Modal Secret:** `hf-secret` (`HF_TOKEN`)
- **Hugging Face Repo:** `gregoreesmaa/sheet-music-converter-omr` (`repo_type="dataset"`)
  - Final Eval JSON: `evals/2026-11-26T00-55-trial-scribe-ttt-suite31-hw.json`
  - Run Ledger JSON: `runs/2026-11-26T00-55-trial-scribe-ttt-suite31-hw.json`

### Copy-Paste Monitoring / Control / Download Commands
```powershell
# Check status of Modal app
.\.venv\Scripts\modal.exe app list

# Tail live logs
.\.venv\Scripts\modal.exe app logs ap-eLJrorKeWReglJ2HRrX7IB

# Stop runaway job if needed
.\.venv\Scripts\modal.exe app stop ap-eLJrorKeWReglJ2HRrX7IB

# Download final eval JSON from Hugging Face once completed
.\.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('gregoreesmaa/sheet-music-converter-omr', 'evals/2026-11-26T00-55-trial-scribe-ttt-suite31-hw.json', repo_type='dataset'))"
```

## 3. Required Data & Baseline Checkpoint (100% Permissive)
- **Eval Dataset:** HF `datasets/suite31_hw_eval.tar` (unpacked to `/vol/datasets/suite31_hw_eval/`), manifest `suite31/suite31_hw_manifest.json` — filtering to all **14 real historical handwritten manuscripts** across both `clean` and `photo` domains (`28` pages total, 100% permissive).
- **Baseline Checkpoint:** HF `checkpoints/iter10_soup_baseline/uni_omr_soup.pt` (cached at `/vol/checkpoints/iter10_soup_baseline/uni_omr_soup.pt`).
  - Measured Iter-10 `soup` baseline (`max_dim=1536`, `max_len=4096`, FSA grammar + MTP-4): `hw_clean 0.2550`, `hw_photo 0.1516`, `printed_clean 0.8895`, `printed_photo 0.8738`, `all45_clean 0.6921`.
- **Scoring Binaries:** Native Linux release builds of `converter/assembly-serialize` and `converter/eval-harness` (linked against sibling `mscz-rs`) compiled in a cached Modal image layer and wired into `eval_sys_gate.ASER` / `eval_sys_gate.EHARNESS` for exact apples-to-apples Strict Note F1 and Note SER.

## 4. Exact Launch Command & Hyperparameters
```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\modal.exe run --detach experiments/2026-11-26T00-55-trial-scribe-ttt-suite31-hw/run_modal.py
```
- **Hyperparameters:**
  - `step_counts = [3, 5]` (each evaluated cleanly from `ttt.restore()` snapshot)
  - `ttt_lr = 5e-5`, `adapt_scope = "norms"` (`RMSNorm` weights in `backbone`/`enc_layers`/`enc_norm`, `< 0.1%` of params)
  - `conf_threshold = 0.85`, `cons_weight = 1.0`, `ent_weight = 1.0`, `kl_weight = 0.5`, `l2_weight = 1e-2`, `grad_clip = 1.0`
  - `max_dim = 1536`, `max_len = 4096`, `ttt_max_decode_len = 512`, `use_grammar = True`, `use_mtp_spec = True`, `bf16 = True`
  - SDPA guard: `enable_cudnn_sdp(False)` (CUTLASS/Flash only).

## 5. 5-Minute Checkpointing & Resume Plan
- Atomic JSON checkpointing (`_atomic_write_json` via `.tmp` → `os.replace`) to `/vol/evals/2026-11-26T00-55-trial-scribe-ttt-suite31-hw_partial.json` + `vol.commit()` every `<= 5.0` minutes (and after every completed piece).
- **Resume:** Re-running the exact same `modal run --detach` command automatically loads `/vol/evals/2026-11-26T00-55-trial-scribe-ttt-suite31-hw_partial.json` and skips already-scored `(piece, domain)` pairs.
- Final eval + run ledger JSONs are uploaded to Hugging Face Hub on completion.

## 6. Evaluation Gate & Decision Tree ("What We Do With the Result")
- **Primary Metrics:** `hw_clean` and `hw_photo` Strict Note F1 (and Token F1) across all 14 HW manuscripts (`before` vs `after_s3` vs `after_s5`), win/tie/loss count per piece, and median TTT wall-clock seconds/page.
- **PROMOTE:** If `delta_note_f1 >= +0.020` on `hw_clean` OR `hw_photo` (with no regression `> -0.005` on the other domain and `wins > losses`) → integrate optional `--ttt-steps` flag into `eval_uni_suite31.py` / inference CLI and run a follow-up full 45-piece check (to verify zero regression on printed scores).
- **ITERATE:** If `delta_note_f1` is modestly positive (`+0.005` to `+0.020`) or shows high variance across manuscripts → inspect per-piece `conf_tokens`, `loss_history`, and `delta_note_f1` in the JSON, and run a targeted follow-up sweeping `adapt_scope="all_rms_norms"` / `"norms_with_gn"`, `ttt_lr`, or `conf_threshold`.
- **DISCARD:** If both `s=3` and `s=5` yield `delta_note_f1 <= +0.002` or regress on `hw_clean` and `hw_photo` → document root cause in `OUTCOME.md` (e.g., too few high-confidence pseudo-tokens on hard HW pages) and shelve SCRIBE/DocTTT until base HW accuracy exceeds `0.40+`.
