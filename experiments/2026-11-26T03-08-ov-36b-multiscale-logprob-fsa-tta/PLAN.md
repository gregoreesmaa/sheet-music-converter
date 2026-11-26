# Experiment Plan — `2026-11-26T03-08-ov-36b-multiscale-logprob-fsa-tta`

## 1. Platform Chosen & Rationale
- **Platform:** **Google Colab `L4` (`24 GB`, `cc=8.9`, native BF16)** via `.\.venv\Scripts\colab.exe` (`~4.8 CU/hr`).
- **Rationale:** 14 real historical HW pieces × 2 domains (`clean`, `photo`) = 28 pages × 5 clean scale views = 140 fast MTP-4 decodes at `max_len=3840` + 140 single-pass teacher-forced log-prob scorings (~30 ms each). Estimated wall-clock: **~12–15 min** (`~1.0–1.2 CU`).

## 2. Identifiers for Follow-Up Queries (Zero-Prior-Context Handoff)
- **Colab Session Name (recommended):** `ov-36b`
- **Hugging Face Repo:** `gregoreesmaa/sheet-music-converter-omr` (`repo_type="dataset"`)
  - Final Eval JSON: `evals/2026-11-26T03-08-ov-36b-multiscale-logprob-fsa-tta.json`
  - Run Ledger JSON: `runs/2026-11-26T03-08-ov-36b-multiscale-logprob-fsa-tta.json`
- **Depends On:** `2026-11-26T01-38-ov-36-hw-multiview-tta-voting` (`COMPLETED`)

### Copy-Paste Colab Launch, Monitor & Cleanup Commands
```powershell
# 1. Provision named L4 session
.\.venv\Scripts\colab.exe new -s ov-36b --gpu L4

# 2. Upload code bundle and temporary HF token (deleted on exit by run_colab.py)
.\.venv\Scripts\colab.exe upload -s ov-36b /path/to/bundle.tar.gz /content/bundle.tar.gz
.\.venv\Scripts\colab.exe upload -s ov-36b /path/to/.hf_token /tmp/.hf_token

# 3. Execute experiment
.\.venv\Scripts\colab.exe exec -s ov-36b -f experiments/2026-11-26T03-08-ov-36b-multiscale-logprob-fsa-tta/run_colab.py

# 4. ALWAYS stop session when finished
.\.venv\Scripts\colab.exe stop -s ov-36b

# 5. Download final eval JSON from Hugging Face
.\.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('gregoreesmaa/sheet-music-converter-omr', 'evals/2026-11-26T03-08-ov-36b-multiscale-logprob-fsa-tta.json', repo_type='dataset'))"
```

## 3. Required Data & Baseline Checkpoint (100% Permissive)
- **Eval Dataset:** HF `datasets/suite31_hw_eval.tar` (`suite31/suite31_hw_manifest.json`) — all **14 real historical handwritten manuscripts** (`clean` + `photo` = 28 pages).
- **Baseline Checkpoint:** HF `checkpoints/iter10_soup_baseline/uni_omr_soup.pt`.
- **Scoring Binaries:** Native Linux release builds of `converter/assembly-serialize` and `converter/eval-harness`.

## 4. Exact TTA Views, Rerankers & Hyperparameters
- **5 Clean In-Distribution Scale Views per Page (NO `clahe`, NO `deskew`, NO OOD `1792px`):**
  1. `scale_1280` (`1280px` — primary strong single-view control from `ov-35`/`ov-36`)
  2. `base_1536` (`1536px` — legacy baseline control)
  3. `scale_1024` (`1024px`)
  4. `scale_1152` (`1152px`)
  5. `scale_1408` (`1408px`)
- **Unsupervised Signals Computed per View:**
  - `fsa_beat_score`: `bekern-decode` validity + EOS completion + per-bar meter & cross-voice beat conservation rate - mean meter/voice error - grammar broken rate.
  - `mean_logprob`: exact length-normalized token log-probability from a single `torch.no_grad()` teacher-forced forward pass `model(img_t, seq[:, :-1], return_aux=False, return_mtp=False)` over `seq[:, 1:]`.
  - `mean_entropy`: exact mean next-token predictive entropy from the same forward pass.
  - `mbr_vote_score`: validity-weighted mean pairwise ordered `compute_token_lcs_f1` across the 5 scale views.
- **Unsupervised Selectors Evaluated (Full 5-View + Fast 2-View `{1280, 1536}` & 3-View `{1152, 1280, 1536}` Subsets):**
  - `best_fsa_beat`: maximizes `fsa_beat_score`.
  - `best_logprob`: maximizes `mean_logprob` (subject to `bekern_decode_ok` + `hit_eos` bonus).
  - `best_neg_entropy`: minimizes `mean_entropy`.
  - `best_fsa_logprob`: `fsa_beat_score + 2.0 * mean_logprob`.
  - `best_mbr_vote`: maximizes `mbr_vote_score`.
  - `best_fsa_logprob_mbr`: `0.5 * fsa_beat_score + 1.5 * mean_logprob + 1.0 * mbr_vote_score`.
  - `oracle_best_view`: upper bound by ground-truth Strict Note F1.
- **Fixed Decode Settings:** `max_len = 3840` (per `ov-35`), `use_grammar = True`, `strict_grammar = False`, `use_mtp_spec = True`, `measure_beam_size = 0`, `bf16 = True`, `enable_cudnn_sdp(False)`.

## 5. 5-Minute Checkpointing & Resume Plan
- Atomic JSON checkpointing (`_atomic_write_json`) to `/content/cache/evals/2026-11-26T03-08-ov-36b-multiscale-logprob-fsa-tta_partial.json` every `<= 5.0` minutes (and after every completed piece).
- Re-running `run_colab.py` automatically resumes from `_partial.json`.
- Uploads final `evals/<exp_id>.json` and `runs/<exp_id>.json` to HF; deletes `/tmp/.hf_token` immediately and in `finally:`.

## 6. Evaluation Gate & Decision Tree ("What We Do With the Result")
- **PROMOTE:** If any unsupervised selector (especially on the fast 2-view `{1280, 1536}` or 3-view `{1152, 1280, 1536}` pool) beats **both** `base_1536` and `scale_1280` by `>= +0.015` on `hw_clean` AND `>= +0.010` on `hw_photo` (with `wins > losses` and zero domain regression) -> wire `--tta-scales` + `--tta-rerank` into `eval_uni_suite31.py` and verify full 45-piece `suite31_hw` parity.
- **ITERATE:** If `mean_logprob` + `fsa_beat` improves over `fsa_beat` alone but still leaves `>= +0.025` oracle gap on `hw_photo` -> test render-cycle visual verifier reranking (scoring re-rendered notation against the input image).
- **DISCARD:** If `mean_logprob` adds `< +0.003` over pure `fsa_beat_score` on clean multi-scale views -> promote the simpler pure `fsa_beat_score` 2-view/3-view scale TTA directly and discard teacher-forced log-prob rescoring.
