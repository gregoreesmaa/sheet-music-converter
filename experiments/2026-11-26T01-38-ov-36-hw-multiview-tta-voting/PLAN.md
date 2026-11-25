# Experiment Plan — `2026-11-26T01-38-ov-36-hw-multiview-tta-voting`

## 1. Platform Chosen & Rationale
- **Platform:** **Google Colab `L4` (`24 GB`, `cc=8.9`, native BF16)** via `.\.venv\Scripts\colab.exe` (`~4.8 CU/hr`).
- **Rationale:** 14 real historical HW pieces x 2 domains (`clean`, `photo`) = 28 pages x 6 views = 168 fast MTP-4 decodes on L4 BF16, plus lightweight CPU preprocessing (fine deskew + CLAHE) and Rust `assembly-serialize` / `eval-harness` scoring. Estimated wall-clock: **~10–15 min** (`~0.8–1.2 CU`).

## 2. Identifiers for Follow-Up Queries (Zero-Prior-Context Handoff)
- **Colab Session Name (recommended):** `ov36-hw-tta-l4`
- **Hugging Face Repo:** `gregoreesmaa/sheet-music-converter-omr` (`repo_type="dataset"`)
  - Final Eval JSON: `evals/2026-11-26T01-38-ov-36-hw-multiview-tta-voting.json`
  - Run Ledger JSON: `runs/2026-11-26T01-38-ov-36-hw-multiview-tta-voting.json`
- **Baseline Commit SHA:** `164852f`

### Copy-Paste Colab Launch, Monitor & Cleanup Commands
```powershell
# 1. Provision named L4 session
.\.venv\Scripts\colab.exe new -s ov36-hw-tta-l4 --gpu L4

# 2. Upload code bundle and temporary HF token (deleted on exit by run_colab.py)
.\.venv\Scripts\colab.exe upload -s ov36-hw-tta-l4 /path/to/bundle.tar.gz /content/bundle.tar.gz
.\.venv\Scripts\colab.exe upload -s ov36-hw-tta-l4 /path/to/.hf_token /tmp/.hf_token

# 3. Execute experiment
.\.venv\Scripts\colab.exe exec -s ov36-hw-tta-l4 -f experiments/2026-11-26T01-38-ov-36-hw-multiview-tta-voting/run_colab.py

# 4. ALWAYS stop session when finished
.\.venv\Scripts\colab.exe stop -s ov36-hw-tta-l4

# 5. Download final eval JSON from Hugging Face
.\.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('gregoreesmaa/sheet-music-converter-omr', 'evals/2026-11-26T01-38-ov-36-hw-multiview-tta-voting.json', repo_type='dataset'))"
```

## 3. Required Data & Baseline Checkpoint (100% Permissive)
- **Eval Dataset:** HF `datasets/suite31_hw_eval.tar` (`suite31/suite31_hw_manifest.json`) — filtered to all **14 real historical handwritten manuscripts** (`clean` + `photo` = 28 pages, 100% permissive).
- **Baseline Checkpoint:** HF `checkpoints/iter10_soup_baseline/uni_omr_soup.pt`.
- **Scoring Binaries:** Native Linux release builds of `converter/assembly-serialize` and `converter/eval-harness` wired into `eval_sys_gate.ASER` / `eval_sys_gate.EHARNESS`.

## 4. Exact TTA Views, Rerankers & Hyperparameters
- **6 Deterministic Views per Page:**
  1. `base_1536`: raw image resized to `1536px` (exact baseline).
  2. `scale_1280`: raw image resized to `1280px`.
  3. `scale_1792`: raw image resized to `1792px`.
  4. `clahe_1536`: LAB-luminance CLAHE (`clipLimit=2.2`, `tileGridSize=(8,8)`) resized to `1536px` (pure-NumPy/PIL fallback if `cv2` absent).
  5. `deskew_1536`: projection-profile fine deskew (`-4.0°` to `+4.0°` in `0.25°` steps maximizing horizontal ink-profile variance, border-filled with median paper tone) resized to `1536px`.
  6. `deskew_clahe_1536`: fine deskew followed by CLAHE resized to `1536px`.
- **4 Unsupervised Selectors (+ Oracle Upper Bound) Evaluated Apples-to-Apples:**
  - `base_1536`: single-view baseline.
  - `best_fsa_beat`: maximizes unsupervised `fsa_beat_score` (`bekern-decode` validity + EOS completion + per-bar meter & cross-voice beat conservation rate via `grammar.PrefixState` - mean meter/voice error - grammar broken rate).
  - `best_mbr_vote`: Minimum Bayes Risk multi-view voting — maximizes validity-weighted mean pairwise ordered `compute_token_lcs_f1` against all other views.
  - `best_combined`: `0.5 * fsa_beat_score + 1.0 * mbr_vote_score`.
  - `oracle_best_view`: best single view by ground-truth Strict Note F1 (measures total TTA headroom).
- **Fixed Decode Settings:** `max_len = 4096`, `use_grammar = True`, `strict_grammar = False`, `use_mtp_spec = True`, `measure_beam_size = 0`, `bf16 = True`, `enable_cudnn_sdp(False)`.

## 5. 5-Minute Checkpointing & Resume Plan
- Atomic JSON checkpointing (`_atomic_write_json`) to `/content/cache/evals/2026-11-26T01-38-ov-36-hw-multiview-tta-voting_partial.json` every `<= 5.0` minutes (and after every completed piece).
- **Resume:** Re-running `run_colab.py` automatically loads `_partial.json` and skips already-completed `(piece, domain)` pairs.
- Uploads final `evals/<exp_id>.json` and `runs/<exp_id>.json` to HF; deletes `/tmp/.hf_token` immediately and in `finally:`.

## 6. Evaluation Gate & Decision Tree ("What We Do With the Result")
- **Primary Metrics:** `hw_clean` and `hw_photo` Strict Note F1 & Token LCS F1 for each individual view, each unsupervised selector (`best_fsa_beat`, `best_mbr_vote`, `best_combined`), and `oracle_best_view`, plus win/tie/loss counts vs `base_1536`.
- **PROMOTE:** If `best_fsa_beat`, `best_mbr_vote`, or `best_combined` improves `hw_clean` OR `hw_photo` by `>= +0.015` Note F1 (with no regression `> -0.005` on the other domain and `wins > losses`) -> wire optional `--tta-views` / `--tta-rerank` into `eval_uni_suite31.py` and the production converter pipeline.
- **ITERATE:** If `oracle_best_view` beats `base_1536` by `>= +0.030` while unsupervised rerankers capture `< 50%` of the headroom -> add model log-prob normalization or render-cycle verifier scoring (from Academic Exp A) to close the reranking gap.
- **DISCARD:** If even `oracle_best_view` adds `< +0.010` over `base_1536` -> document that preprocessing/scale TTA is bottlenecked by base HW representation rather than input view, and prioritize HW training data / architecture upgrades.
