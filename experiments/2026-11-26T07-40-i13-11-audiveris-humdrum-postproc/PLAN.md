# Experiment Plan — `2026-11-26T07-40-i13-11-audiveris-humdrum-postproc`

## 1. Competitor Mechanism & Hypothesis
- **Audiveris (`HEADERS`, `MEASURES`, `RHYTHMS`):** Enforces strict staff header grammar (`Clef -> Key -> Time`), propagates active headers across systems, and checks every measure's cumulative duration against the active time signature ("pink measure" rhythm check).
- **Humdrum `**kern` / Verovio Post-Processing:** Validates spine/voice consistency, normalizes accidental scopes against active key signatures and barline resets, and repairs rhythmic summation drift before engraving.
- **Hypothesis:** Applying deterministic, conservative symbolic repair rules *after* MTP-4+FSA decoding (0 ms GPU overhead, < 2 ms CPU/page) improves Note F1 and `r_rhythm` on `suite31_hw`—particularly eliminating whole-system clef/key drift on multi-system `hw` pages and single-beat duration slips—without the latency or log-prob degradation of online beam search.

## 2. Platform & Identifiers
- **Platform:** Google Colab `L4` (`24 GB`, `cc=8.9`, BF16 native), est. **~12 min** (`~1.0 CU`).
- **HF Repo:** `gregoreesmaa/sheet-music-converter-omr` (`repo_type="dataset"`).
- **Eval Artifact:** `evals/2026-11-26T07-40-i13-11-audiveris-humdrum-postproc.json`.
- **Run Ledger:** `runs/2026-11-26T07-40-i13-11-audiveris-humdrum-postproc.json`.
- **Checkpoint:** Defaults to `checkpoints/iter10_soup_baseline/uni_omr_soup.pt` (overrideable via `HF_CKPT_PATH` env var for Iter 12 `UniOMR v2`).
- **Dataset:** `datasets/suite31_hw_eval.tar` (all 45 pieces x `clean`/`photo` = 90 pages).

## 3. Single-Variable Ablation Design
Run **1 GPU decode pass** (`max_dim=1280, max_len=3840`, MTP-4 + FSA grammar, BF16) across all 90 pages, caching raw `pred_raw` tokens to disk, then evaluate 5 deterministic CPU symbolic arms on the exact same streams:
1. `raw_fsa_baseline`: Unmodified `stitch_uni_tokens(pred_raw)`.
2. `audiveris_header_continuity`: Splits `pred_raw` by `<sys>`. Computes page-level consensus `(clef, key)` from systems that emit explicit headers; if a continuation system omits `clef-*` or `key*` (or emits a 1-system singleton clef outlier on a >=3-system page while surrounded by identical clefs), injects/restores the consensus header token.
3. `humdrum_accidental_norm`: Walks each measure per voice; strips duplicate/malformed double alterations on a single notehead and cleans orphan alteration tokens not attached to a valid pitch.
4. `audiveris_rhythm_repair`: Parses active `time-B-T` (`beats = B * 4 / T`). Within each measure and voice, sums written durations (including `dot` = `1.5x`). If total duration differs from `beats` by an exact single-edit fix on the last note/rest of the measure (adding/removing one `dot`, or halving/doubling the final duration token), applies the exact duration repair.
5. `combined_all_rules`: Applies `header_continuity -> accidental_norm -> rhythm_repair` sequentially.

## 4. Checkpointing & Safety
- Atomic `_partial.json` saved every `<= 5.0` min during the single GPU decode pass; auto-resumes already-decoded pages.
- `/tmp/.hf_token` read with `chmod 600` and deleted immediately and in `finally:`.
- `torch.backends.cuda.enable_cudnn_sdp(False)` enforced.

## 5. Promotion / Discard Gate
- **PROMOTE:** Any individual rule (or `combined_all_rules`) achieves `>= +0.003` Note F1 on `all45_clean` OR `all45_photo` with **zero slice regressing `> -0.001`** -> port winning rule(s) into `converter/assembly-serialize` (Rust) and `eval_uni_suite31.py`.
- **ITERATE:** One rule wins (`>= +0.003`) while another regresses -> promote only the winning rule(s) and refine the regressing rule's guard threshold.
- **DISCARD:** All rules within `[-0.001, +0.002]` with no meaningful `r_rhythm` gain -> close deterministic token-level post-processing.
