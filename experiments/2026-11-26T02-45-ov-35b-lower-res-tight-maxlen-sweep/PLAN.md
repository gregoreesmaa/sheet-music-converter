# Experiment Plan — `2026-11-26T02-45-ov-35b-lower-res-tight-maxlen-sweep`

## 1. Platform Chosen & Rationale
- **Platform:** **Google Colab `L4` (`24 GB`, `cc=8.9`, native BF16)** via `.\.venv\Scripts\colab.exe` (`~4.8 CU/hr`).
- **Rationale:** Deterministic greedy + MTP-4 + FSA-grammar decoding at `max_len=3840` produces the exact token prefix of `2560`, `3072`, and `3584`. Only **4 GPU decode passes** (`1024px`, `1152px`, `1280px`, `1408px` at horizon `3840`) are needed to evaluate all **4 × 4 = 16** `(max_dim, max_len)` cells with bit-exact parity. Because horizon `3840` avoids the long `6144`-token HW runaway loops from `ov-35`, estimated wall-clock is **~14–16 min** (`~1.2 CU`).

## 2. Identifiers for Follow-Up Queries (Zero-Prior-Context Handoff)
- **Colab Session Name (recommended):** `ov-35b`
- **Hugging Face Repo:** `gregoreesmaa/sheet-music-converter-omr` (`repo_type="dataset"`)
  - Final Eval JSON: `evals/2026-11-26T02-45-ov-35b-lower-res-tight-maxlen-sweep.json`
  - Run Ledger JSON: `runs/2026-11-26T02-45-ov-35b-lower-res-tight-maxlen-sweep.json`
- **Predecessor Experiment:** `2026-11-26T01-38-ov-35-maxlen-resolution-inference-sweep` (`COMPLETED`, `ITERATE`)

## 3. Required Data & Baseline Checkpoint (100% Permissive)
- **Eval Dataset:** HF `datasets/suite31_hw_eval.tar` (`suite31/suite31_hw_manifest.json`) — all **45 pieces** (31 printed + 14 real historical HW, `clean` + `photo` = 90 pages).
- **Baseline Checkpoint:** HF `checkpoints/iter10_soup_baseline/uni_omr_soup.pt` (apples-to-apples with `ov-35`).
- **Scoring Binaries:** Native Linux release builds of `converter/assembly-serialize` and `converter/eval-harness`.

## 4. Exact Sweep Grid & Hyperparameters
- **Grid (16 cells from 4 GPU passes):**
  - `max_dim in {1024, 1152, 1280, 1408}`
  - `max_len in {2560, 3072, 3584, 3840}`
- **Fixed Inference Settings:**
  - `use_grammar = True`, `strict_grammar = False`, `use_mtp_spec = True`, `measure_beam_size = 0`, `bf16 = True`
  - SDPA guard: `torch.backends.cuda.enable_cudnn_sdp(False)` (CUTLASS/Flash only).

## 5. 5-Minute Checkpointing & Resume Plan
- Atomic JSON checkpointing to `/content/cache/evals/2026-11-26T02-45-ov-35b-lower-res-tight-maxlen-sweep_partial.json` every `<= 5.0` minutes and after each resolution pass.
- Uploads final `evals/<exp_id>.json` and `runs/<exp_id>.json` to HF; deletes `/tmp/.hf_token` immediately and in `finally:`.

## 6. Evaluation Gate & Decision Tree ("What We Do With the Result")
- **PROMOTE:** Lock the winning interior `(max_dim, max_len)` cell (e.g., `1280px / 3072–3840` or `1152px` / `1408px`) as the new default inference config in `eval_uni_suite31.py`, `docs/UNIOMR.md`, and Iter 11 eval scripts if it clearly brackets the peak (strictly beats both neighbors) without regressing `hw_photo` by `> 0.005`.
- **ITERATE:** If `1280px` wins `clean` while `1408px` wins `hw_photo`, wire a lightweight two-scale (`1280px + 1408px/1536px`) FSA/beat-conservation reranker into the production HW inference path (combining with `ov-36` TTA findings).
- **DISCARD:** If `1024px` or `2560` tokens degrade printed/HW accuracy significantly, record the exact lower cliff in `OUTCOME.md` and promote `(1280, 3840)` from `ov-35`.
