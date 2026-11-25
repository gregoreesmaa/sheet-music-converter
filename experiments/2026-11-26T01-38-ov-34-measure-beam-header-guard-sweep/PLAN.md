# Experiment Plan — `2026-11-26T01-38-ov-34-measure-beam-header-guard-sweep`

## 1. Platform Chosen & Rationale
- **Platform:** **Google Colab `L4` (`24 GB`, `cc=8.9`, native BF16)** via `.\.venv\Scripts\colab.exe` (`~4.8 CU/hr`).
- **Rationale:** Pure inference-time decoding sweep with zero training gradients; `L4` runs MTP-4 + BF16 full-page decoding rapidly with `torch 2.14.1+cu130` pre-installed. Deduplicating `measure_beam_size=1` (shared greedy+MTP-4 baseline across all `const_offset_limit` values) and reusing cached Rust reference notes + skipping re-scoring when `measure_beam_repaired == 0` keeps total wall-clock to **~18–24 min** (`~1.5–1.9 CU`).

## 2. Identifiers for Follow-Up Queries (Zero-Prior-Context Handoff)
- **Colab Session Name (recommended):** `ov34-mbeam-l4`
- **Hugging Face Repo:** `gregoreesmaa/sheet-music-converter-omr` (`repo_type="dataset"`)
  - Final Eval JSON: `evals/2026-11-26T01-38-ov-34-measure-beam-header-guard-sweep.json`
  - Run Ledger JSON: `runs/2026-11-26T01-38-ov-34-measure-beam-header-guard-sweep.json`
- **Baseline Commit SHA:** `164852f`

### Copy-Paste Colab Launch, Monitor & Cleanup Commands
```powershell
# 1. Provision named L4 session
.\.venv\Scripts\colab.exe new -s ov34-mbeam-l4 --gpu L4

# 2. Upload code bundle and temporary HF token (deleted on exit by run_colab.py)
.\.venv\Scripts\colab.exe upload -s ov34-mbeam-l4 /path/to/bundle.tar.gz /content/bundle.tar.gz
.\.venv\Scripts\colab.exe upload -s ov34-mbeam-l4 /path/to/.hf_token /tmp/.hf_token

# 3. Execute experiment
.\.venv\Scripts\colab.exe exec -s ov34-mbeam-l4 -f experiments/2026-11-26T01-38-ov-34-measure-beam-header-guard-sweep/run_colab.py

# 4. ALWAYS stop session when finished (zero idle CU burn)
.\.venv\Scripts\colab.exe stop -s ov34-mbeam-l4

# 5. Download final eval JSON from Hugging Face
.\.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('gregoreesmaa/sheet-music-converter-omr', 'evals/2026-11-26T01-38-ov-34-measure-beam-header-guard-sweep.json', repo_type='dataset'))"
```

## 3. Required Data & Baseline Checkpoint (100% Permissive)
- **Eval Dataset:** HF `datasets/suite31_hw_eval.tar` (unpacked to `/content/cache/datasets/suite31_hw_eval/`), manifest `suite31/suite31_hw_manifest.json` — all **45 pieces** (31 printed + 14 real historical HW, `clean` + `photo` = 90 pages, 100% permissive).
- **Baseline Checkpoint:** HF `checkpoints/iter10_soup_baseline/uni_omr_soup.pt`.
- **Scoring Binaries:** Native Linux release builds of `converter/assembly-serialize` and `converter/eval-harness` (linked against `mscz-rs`) wired into `eval_sys_gate.ASER` / `eval_sys_gate.EHARNESS`.

## 4. Exact Sweep Grid & Hyperparameters
- **Grid (10 unique configurations):**
  - Baseline: `(measure_beam_size=1, const_offset_limit=0)` — greedy + MTP-4 + FSA grammar (`use_measure_beam=False`).
  - Sweep: `measure_beam_size in {2, 4, 8}` x `const_offset_limit in {2, 3, 5}` (9 guarded beam configs).
- **Fixed Inference Settings:**
  - `max_dim = 1536`, `max_len = 4096`, `max_bar_toks = 120`
  - `use_grammar = True`, `strict_grammar = False`, `use_mtp_spec = True`, `bf16 = True`
  - SDPA guard: `torch.backends.cuda.enable_cudnn_sdp(False)` (CUTLASS/Flash only).
- **Mechanism:** `run_colab.py` temporarily binds `eval_uni_suite31.MeasureBeamHeaderGuard` via a context manager so each sweep cell instantiates `MeasureBeamHeaderGuard(const_offset_limit=limit)` cleanly without mutating repo source files.

## 5. 5-Minute Checkpointing & Resume Plan
- Atomic JSON checkpointing (`_atomic_write_json` via `.tmp` -> `os.replace`) to `/content/cache/evals/2026-11-26T01-38-ov-34-measure-beam-header-guard-sweep_partial.json` every `<= 5.0` minutes (and after every completed configuration).
- **Resume:** Re-running `run_colab.py` in the same session automatically loads `_partial.json` and skips already-completed `(measure_beam_size, const_offset_limit)` configs.
- Final `evals/<exp_id>.json` and `runs/<exp_id>.json` are uploaded to Hugging Face Hub; `/tmp/.hf_token` is purged immediately after reading and again in `finally:`.

## 6. Evaluation Gate & Decision Tree ("What We Do With the Result")
- **Primary Metrics:** `printed_clean`, `printed_photo`, `hw_clean`, `hw_photo`, `all45_clean`, `all45_photo` Strict Note F1, Token LCS F1, decode-OK rate, total triggers/repairs, header-guard trip rate, and runtime overhead (`elapsed_secs` vs `mbeam=1` baseline).
- **PROMOTE:** If any `(measure_beam_size, const_offset_limit)` config achieves `>= +0.005` Strict Note F1 on `all45_clean` OR `all45_photo` with zero slice regressing `> -0.002` and `< 25%` runtime overhead -> expose `--header-guard-limit` CLI flag in `eval_uni_suite31.py` and set the winning `(measure_beam_size, const_offset_limit)` as the recommended production decode preset.
- **ITERATE:** If `all45` gain is positive (`+0.002` to `+0.005`) or helps only one domain (e.g., printed vs HW) -> inspect per-piece `measure_beam_triggers`, `measure_beam_repaired`, and `header_guard_tripped` in the JSON to refine the penalty weights (`6.0 * meter_err + 4.0 * voice_err`) or domain-gate measure beam.
- **DISCARD:** If all guarded configs produce `<= +0.001` net Note F1 or regress any slice `> -0.003` -> keep default `measure_beam_size=0` (pure MTP-4 + FSA grammar) and document findings in `OUTCOME.md`.
