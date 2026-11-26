# Experiment Plan — `2026-11-26T07-40-i13-13-cross-system-header-consensus`

## 1. Platform Chosen & Rationale
- **Platform:** **Google Colab `L4` (`24 GB`, `cc=8.9`, native BF16)** via `.\.venv\Scripts\colab.exe` (`~4.8 CU/hr`).
- **Rationale:** Pure inference-time decoding sweep across 4 arms on 90 `suite31_hw` pages at canonical `1280px / 3840`. Estimated wall-clock: **~22–28 min** (`~2.0–2.2 CU`).

## 2. Identifiers for Follow-Up Queries
- **Colab Session Name (recommended):** `i13-13-hdr-cons-l4`
- **Hugging Face Repo:** `gregoreesmaa/sheet-music-converter-omr` (`repo_type="dataset"`)
  - Final Eval JSON: `evals/2026-11-26T07-40-i13-13-cross-system-header-consensus.json`
  - Run Ledger JSON: `runs/2026-11-26T07-40-i13-13-cross-system-header-consensus.json`
- **Depends On:** None (runs immediately against `checkpoints/iter10_soup_baseline/uni_omr_soup.pt`, and supports `HF_CKPT_PATH` env override for `UniOMR v2` checkpoints).

## 3. Required Data & Checkpoint (100% Permissive)
- **Eval Dataset:** HF `datasets/suite31_hw_eval.tar` -> `suite31/suite31_hw_manifest.json` (45 pieces x `clean`/`photo` = 90 pages).
- **Checkpoint:** HF `checkpoints/iter10_soup_baseline/uni_omr_soup.pt` (or `HF_CKPT_PATH` override).
- **Scoring Binaries:** Native Linux release builds of `assembly-serialize` and `eval-harness`.

## 4. Exact Sweep Grid & Mechanism
- **Fixed Inference Settings:** `max_dim=1280`, `max_len=3840`, `use_grammar=True`, `strict_grammar=False`, `bf16=True`, `torch.backends.cuda.enable_cudnn_sdp(False)`.
- **4 Arms:**
  1. `baseline`: Standard `ev.generate_mtp_spec` (no cross-system header propagation).
  2. `online_sys1_bias_1.5_c85`: During KV-cached constrained decode, record System 1's leading header tokens (`clef-*`, `key*`, `time-*`). Immediately after each `<sys>` token, while still in the header phase, if top-1 softmax probability `< 0.85`, add `+1.5` logit bias to the next expected System-1 header token.
  3. `online_sys1_bias_3.0_c90`: Same mechanism with `+3.0` logit bias and `< 0.90` confidence gate.
  4. `twopass_majority_header`: Compute teacher-forced token probabilities on the baseline decode; take confidence-weighted page-wide majority `(clef_seq, key)` across all systems; for up to 3 outlier continuation systems whose header disagrees with majority AND has `mean_header_prob < 0.85`, re-decode from that `<sys>` prefix with the consensus header forced and accept if `fsa_beat_score` does not degrade.
- **Implementation:** Self-contained in `run_colab.py` without mutating `uni_omr_v2.py` or `train_uni_v2.py`.

## 5. 5-Minute Checkpointing & Resume
- Atomic JSON checkpointing (`_partial.json` via `.tmp` -> `os.replace`) every `<= 5.0` min and after every completed arm; auto-resumes completed arms.

## 6. Evaluation Gate & Decision Tree
- **PROMOTE:** Any arm achieves `>= +0.005` Strict Note F1 on `all45_photo` OR `hw_photo` (or `>= +0.004` on `all45_clean`), with zero slice regressing `> -0.002` and `< 20%` latency overhead -> wire `--cross-sys-header-mode` into `eval_uni_suite31.py`.
- **ITERATE:** Gains appear on `photo` but mid-piece key/clef changes regress -> tighten confidence gate (`< 0.70`) or restrict bias to `clef-*` + `key*` only (excluding `time-*`).
- **DISCARD:** All arms within `[-0.002, +0.002]` across all slices -> close cross-system header propagation line.
