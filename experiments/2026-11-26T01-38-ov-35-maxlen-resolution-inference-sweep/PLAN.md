# Experiment Plan — `2026-11-26T01-38-ov-35-maxlen-resolution-inference-sweep`

## 1. Platform Chosen & Rationale
- **Platform:** **Google Colab `L4` (`24 GB`, `cc=8.9`, native BF16)** via `.\.venv\Scripts\colab.exe` (`~4.8 CU/hr`).
- **Rationale:** Because deterministic greedy + MTP-4 + FSA-grammar decoding at `max_len=6144` produces the **exact** token prefix of `max_len=3840` and `max_len=5120` for any fixed input resolution `max_dim`, we only need **3 GPU decode passes** (`1280px`, `1536px`, `1792px` at `max_len=6144`) to evaluate all **3 x 3 = 9** `(max_dim, max_len)` cells with bit-exact parity (applying `trim_truncated_tokens` whenever `eos_id` is not reached within the cutoff). Estimated wall-clock: **~12–16 min** (`~1.0–1.3 CU`).

## 2. Identifiers for Follow-Up Queries (Zero-Prior-Context Handoff)
- **Colab Session Name (recommended):** `ov35-maxlen-res-l4`
- **Hugging Face Repo:** `gregoreesmaa/sheet-music-converter-omr` (`repo_type="dataset"`)
  - Final Eval JSON: `evals/2026-11-26T01-38-ov-35-maxlen-resolution-inference-sweep.json`
  - Run Ledger JSON: `runs/2026-11-26T01-38-ov-35-maxlen-resolution-inference-sweep.json`
- **Baseline Commit SHA:** `164852f`

### Copy-Paste Colab Launch, Monitor & Cleanup Commands
```powershell
# 1. Provision named L4 session
.\.venv\Scripts\colab.exe new -s ov35-maxlen-res-l4 --gpu L4

# 2. Upload code bundle and temporary HF token (deleted on exit by run_colab.py)
.\.venv\Scripts\colab.exe upload -s ov35-maxlen-res-l4 /path/to/bundle.tar.gz /content/bundle.tar.gz
.\.venv\Scripts\colab.exe upload -s ov35-maxlen-res-l4 /path/to/.hf_token /tmp/.hf_token

# 3. Execute experiment
.\.venv\Scripts\colab.exe exec -s ov35-maxlen-res-l4 -f experiments/2026-11-26T01-38-ov-35-maxlen-resolution-inference-sweep/run_colab.py

# 4. ALWAYS stop session when finished
.\.venv\Scripts\colab.exe stop -s ov35-maxlen-res-l4

# 5. Download final eval JSON from Hugging Face
.\.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('gregoreesmaa/sheet-music-converter-omr', 'evals/2026-11-26T01-38-ov-35-maxlen-resolution-inference-sweep.json', repo_type='dataset'))"
```

## 3. Required Data & Baseline Checkpoint (100% Permissive)
- **Eval Dataset:** HF `datasets/suite31_hw_eval.tar` (`suite31/suite31_hw_manifest.json`) — all **45 pieces** (31 printed + 14 real historical HW, `clean` + `photo` = 90 pages, 100% permissive).
- **Baseline Checkpoint:** HF `checkpoints/iter10_soup_baseline/uni_omr_soup.pt`.
- **Scoring Binaries:** Native Linux release builds of `converter/assembly-serialize` and `converter/eval-harness` wired into `eval_sys_gate.ASER` / `eval_sys_gate.EHARNESS`.

## 4. Exact Sweep Grid & Hyperparameters
- **Grid (9 cells from 3 GPU passes):**
  - `max_dim in {1280, 1536, 1792}`
  - `max_len in {3840, 5120, 6144}`
- **Fixed Inference Settings:**
  - `use_grammar = True`, `strict_grammar = False`, `use_mtp_spec = True`, `measure_beam_size = 0`, `bf16 = True`
  - SDPA guard: `torch.backends.cuda.enable_cudnn_sdp(False)` (CUTLASS/Flash only).
- **Diagnostics Recorded per Page:** `hit_eos_6144`, `raw_gen_len`, `truncated_at_{3840,5120,6144}`, `elapsed_secs`, `tok_per_sec`, `tok_f1`, `tok_lcs`, `note_f1`, `note_ser`, `decode_ok`, and `n_sys`.

## 5. 5-Minute Checkpointing & Resume Plan
- Atomic JSON checkpointing (`_atomic_write_json`) to `/content/cache/evals/2026-11-26T01-38-ov-35-maxlen-resolution-inference-sweep_partial.json` every `<= 5.0` minutes and after each resolution pass (`1280`, `1536`, `1792`).
- **Resume:** Re-running `run_colab.py` automatically loads `_partial.json` and skips completed resolutions.
- Uploads final `evals/<exp_id>.json` and `runs/<exp_id>.json` to HF; deletes `/tmp/.hf_token` immediately and in `finally:`.

## 6. Evaluation Gate & Decision Tree ("What We Do With the Result")
- **Primary Metrics:** 3 x 3 Strict Note F1 matrix across `printed_clean`, `printed_photo`, `hw_clean`, `hw_photo`, `all45_clean`, `all45_photo`, plus truncation rate per `max_len` and wall-clock latency per `max_dim`.
- **PROMOTE:** If any `(max_dim, max_len)` cell beats `(1536, 3840)` by `>= +0.005` on `all45_clean` or `all45_photo` (or `>= +0.015` on `hw_clean`/`hw_photo` without printed regression) -> update default inference/eval `--max-dim` and `--max-len` in `eval_uni_suite31.py` and `docs/UNIOMR.md`.
- **ITERATE:** If `1792px` helps printed fine-detail pieces while `1280px` helps HW pieces (or vice-versa) -> feed directly into Exp 36 multi-scale TTA and adaptive resolution routing (e.g., dense-page upscaling).
- **DISCARD:** If `(1536, 3840/4096)` already sits on the Pareto frontier with `< +0.002` gain from `1792px` or `6144` tokens -> lock `1536px / 4096` as optimal and record the exact truncation ceiling in `OUTCOME.md`.
