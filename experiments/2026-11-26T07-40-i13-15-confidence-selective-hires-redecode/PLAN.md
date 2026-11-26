# Experiment Plan — `2026-11-26T07-40-i13-15-confidence-selective-hires-redecode`

## 1. Platform Chosen & Rationale
- **Platform:** **Google Colab `L4` (`24 GB`, `cc=8.9`, native BF16)** via `.\.venv\Scripts\colab.exe` (`~4.8 CU/hr`).
- **Rationale:** Pass 1 runs `1280px` once across all 90 pages (reused by all triage arms!), and Pass 2 selectively runs `1152px`/`1536px` only on flagged pages/systems. Estimated wall-clock: **~25–32 min** (`~2.1–2.6 CU`).

## 2. Identifiers for Follow-Up Queries
- **Colab Session Name (recommended):** `i13-15-conf-triage-l4`
- **Hugging Face Repo:** `gregoreesmaa/sheet-music-converter-omr` (`repo_type="dataset"`)
  - Final Eval JSON: `evals/2026-11-26T07-40-i13-15-confidence-selective-hires-redecode.json`
  - Run Ledger JSON: `runs/2026-11-26T07-40-i13-15-confidence-selective-hires-redecode.json`
- **Depends On:** None (runs immediately against `checkpoints/iter10_soup_baseline/uni_omr_soup.pt`, and supports `HF_CKPT_PATH` env override for `UniOMR v2` checkpoints).

## 3. Required Data & Checkpoint (100% Permissive)
- **Eval Dataset:** HF `datasets/suite31_hw_eval.tar` -> `suite31/suite31_hw_manifest.json` (45 pieces x `clean`/`photo` = 90 pages).
- **Checkpoint:** HF `checkpoints/iter10_soup_baseline/uni_omr_soup.pt` (or `HF_CKPT_PATH` override).
- **Scoring Binaries:** Native Linux release builds of `assembly-serialize` and `eval-harness`.

## 4. Exact Sweep Grid & Mechanism
- **Fixed Inference Settings:** `max_dim=1280`, `max_len=3840`, `use_grammar=True`, `strict_grammar=False`, `use_mtp_spec=True`, `bf16=True`, `torch.backends.cuda.enable_cudnn_sdp(False)`.
- **4 Arms + Calibration Audit:**
  1. `baseline_1280`: Single-pass `1280px` + exact teacher-forced token log-probs & per-system confidence stats (`~30 ms/page`).
  2. `conf_triage_page_p25`: Trigger 3-scale `{1152, 1280, 1536}` `best_fsa_beat + 0.5 * mean_logprob` reranking ONLY if `mean_logprob < -0.18` OR `beat_ok_rate < 0.85` OR `has_loop` OR `not bekern_decode_ok`.
  3. `conf_triage_page_p40`: Same 3-scale selective reranking with broader gate (`mean_logprob < -0.12` OR `beat_ok_rate < 0.92` OR `has_loop` OR `not bekern_decode_ok`).
  4. `conf_triage_system_hires`: Surgical system-level `1536px` re-decode — for pages with flagged systems (`sys_mean_lp < -0.20` OR `sys_meter_err > 0.01`, capped at `<= 4` systems/page), encode once at `1536px`, prefill KV-cache with the `1280px` prefix up to each flagged `<sys>` boundary, decode ONLY that system, and splice it back in if its system-level `fsa_beat + 0.5 * sys_mean_lp` improves.
  5. **Calibration Audit:** Report exact AUROC of `-mean_logprob`, `1 - beat_ok_rate`, and combined risk score for predicting `note_f1 < 0.85` across all 90 pages.

## 5. 5-Minute Checkpointing & Resume
- Atomic JSON checkpointing (`_partial.json` via `.tmp` -> `os.replace`) every `<= 5.0` min and after every completed stage; auto-resumes completed pages/arms.

## 6. Evaluation Gate & Decision Tree
- **PROMOTE:** Any selective triage arm achieves `>= +0.006` Strict Note F1 on `all45_photo` OR `>= +0.012` on `hw_clean`/`hw_photo` with `< 50%` average latency overhead and zero slice regressing `> -0.002` -> wire `--conf-triage-tta` / `--conf-triage-sys-hires` into `eval_uni_suite31.py`.
- **ITERATE:** Page-level triage wins over system-level splice (or vice versa) -> tune threshold on the winning granularity and expose confidence score in converter output for Soundslice-style suspect-measure UX highlighting.
- **DISCARD:** Selective triage fails to beat `baseline_1280` by `>= +0.003` or triggers on `> 75%` of pages -> keep explicit opt-in `--tta-scales`.
