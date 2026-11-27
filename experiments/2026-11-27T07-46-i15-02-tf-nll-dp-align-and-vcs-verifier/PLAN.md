# PLAN — `2026-11-27T07-46-i15-02-tf-nll-dp-align-and-vcs-verifier`

## 1. Platform & Budget
- **Platform:** Google Colab `1× NVIDIA L4` (`24 GB` VRAM, native `BF16`; auto-falls back to `FP16` on `cc < 8.0` GPUs, never silent `FP32`).
- **Estimated Wall-Clock & Cost:** `~30 min` (`~2.0 CU`).
- **Baseline Commit SHA:** `e426c28`.
- **License Compliance (100% Permissive):** PDM 1.0 archival manuscripts (`suite31_hw`), MIT `hw-score-sim`, MIT/Apache `mscz-rs` / `bekern`. **Zero** `CC-BY-NC`, **zero** GPL/AGPL tools (no MuseScore binary dependency in the verifier; synthetic visual silhouettes and renders are generated deterministically in pure Python/NumPy/PIL).

## 2. Identifiers for Follow-Up Queries
- **Queue Entry:** `experiments/2026-11-27T07-46-i15-02-tf-nll-dp-align-and-vcs-verifier/queue_entry.json` (`id: "2026-11-27T07-46-i15-02-tf-nll-dp-align-and-vcs-verifier"`, `short_id: "i15-02"`, `wave: 13`, `provider: "colab_l4"`, `gpu: "L4"`).
- **Hugging Face (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-27T07-46-i15-02-tf-nll-dp-align-and-vcs-verifier.json`
  - Run Ledger: `runs/2026-11-27T07-46-i15-02-tf-nll-dp-align-and-vcs-verifier.json`
- **Live Cloud IDs:** Recorded on launch by the overnight queue dispatcher (`session`, `started_at`, `log_path`).

## 3. Data & Frozen Checkpoint Protocol
- **Dataset:** `datasets/suite31_hw_eval.tar` (`suite31_hw_manifest.json` — 45 pieces total; Parts A & B evaluate all **14 real historical HW pieces × `{clean, photo}` = 28 HW pages**, and the 2-page fail-fast sanity gate runs on the first 2 printed clean pages).
- **Frozen Checkpoint Fallback Chain:**
  1. `checkpoints/iter12_v2_best/uni_omr_v2_ema_best.pt`
  2. `checkpoints/iter11_v2_best/uni_omr_v2_ema_best.pt`
  3. `checkpoints/iter10_5_v2_best/uni_omr_v2_ema_best.pt`
  4. `checkpoints/iter10_v2_best/uni_omr_v2_ema_best.pt`
  5. `checkpoints/iter10_soup_baseline/uni_omr_soup.pt`
- **Harness Integrity Guards (Mandatory):**
  1. Canonical loader `eval_uni_suite31.load_uni_checkpoint(ckpt_path, device)` with hard `assert len(missing_keys) == 0`.
  2. Canonical `[0, 1]` CHW input normalization via `uni_omr._resize_fullpage_pil` + `uni_omr._pil_to_chw_tensor`.
  3. Native `BF16` on `cc >= 8` (`L4`/`A100`), `FP16` on `cc < 8` (`T4`).
  4. **2-page fail-fast sanity gate** on the first 2 printed clean pages (aborts in `< 30 s` if `dec_ok == 0` or `mean_note_f1 < 0.05`).

## 4. Experimental Protocol (Pivot Strategist 2 — Pivots 1 & 3)

### Part A — `part_a_vcs_verifier_corr` (Visual Consistency Score Verifier)
Evaluates all 14 `suite31_hw` HW pieces × `{clean, photo}` = 28 pages:
1. Decode each HW page at `1536px` (`base_1536`) and `1280px` (`scale_1280`), recording `mean_logprob`, `beat_valid_frac` (`r_rhythm`), `r_parse`, `r_noloop`, and ground-truth **Strict Note F1** against the piece's GT MusicXML.
2. Render a deterministic pure-Python/NumPy/PIL **synthetic visual notation map & page render** from the `1536px` predicted bekern tokens (systems mapped to detected vertical bands, barlines placed proportionally by measure counts, noteheads/stems/beams rendered onto staves).
3. Compute the 4 **Visual Consistency Score (`VCS`)** components in `[0, 1]`:
   - `vcs_ink_iou`: High-pass ink IoU between the real HW page's local-background-subtracted ink map and the predicted bekern's 2D ink map at `(64, 64)` grid resolution.
   - `vcs_barline_x_corr`: Normalized correlation between the real page's vertical-stroke barline-X column projection (within staff bands) and the predicted bekern's barline-X profile.
   - `vcs_cross_scale_agree`: Pitch+duration Note F1 agreement between the `1280px` and `1536px` decodes.
   - `vcs_encoder_patch_cos`: Cosine similarity between frozen UniOMR encoder memory features (global + 4×4 spatial pooled grid) of the real HW page vs the synthetic render of the predicted bekern.
   - Composite `VCS = 0.25 * (vcs_ink_iou + vcs_barline_x_corr + vcs_cross_scale_agree + vcs_encoder_patch_cos)`.
4. **Metrics & `ov-39` Hallucination Rejection:**
   - `pearson_r_vcs_vs_note_f1` & `spearman_rho_vcs_vs_note_f1` across all 28 HW pages (plus per-component `r` and baseline `pearson_r_ov39_logprob_vs_note_f1`).
   - `ov39_hallucination_fp_rejection_rate`: False-positive rejection rate on `ov-39`-passing RISM-like hallucinations (both naturally occurring low-F1 `ov-39`-passing HW decodes AND controlled cross-piece/shifted fluent high-rhythm hallucination negatives passing `beat_valid_frac >= 0.70`, `r_parse >= 0.5`, `r_noloop == 1.0`), plus hallucination-vs-valid AUROC.
- **Gate (Part A):** `Pearson r(VCS, Strict Note F1) >= 0.70`.

### Part B — `part_b_tf_nll_dp_align_bench` (AR-Free Teacher-Forced NLL + Cross-Attention Monotonicity DP)
Evaluates all 14 `suite31_hw` HW pieces × `{clean, photo}` = 28 multi-system HW pages:
1. Parse each piece's GT MusicXML into canonical measure-grouped bekern (`M = [m_0, ..., m_{K-1}]`), zero OOV tokens, and embed the true page span `[m_a..m_b]` inside a multi-page movement stream of `>= 16` measures (simulating alignment against unsegmented full-movement gold XMLs).
2. Construct candidate measure windows around `[m_a..m_b]`:
   - True span `[m_a, m_b]`
   - Shifted windows: `[-2, -2]`, `[-1, -1]`, `[+1, +1]`, `[+2, +2]`
   - Boundary-perturbed windows: `[m_a, m_b - 1]`, `[m_a, m_b + 1]`, `[m_a + 1, m_b]`, `[m_a - 1, m_b]`
   - Off-page distractor windows from elsewhere in the movement stream.
3. Compare 3 alignment methods on the exact same candidate set:
   - **`free_decoded_align` (Baseline):** Score each candidate window `[s..e]` by Note F1 against the model's free-decoded `1536px` prediction.
   - **`tf_nll_only`:** Single encoder pass + parallel teacher-forced decoder pass per candidate (`-mean_ce(bekern[s..e] | image)`), zero AR decoding.
   - **`tf_nll_plus_ca_mono_dp`:** Teacher-forced NLL + **Cross-Attention Monotonicity DP** extracted via non-invasive forward pre-hooks on the last 2 decoder cross-attention layers, projecting per-measure attention centroids onto normalized 2D visual coordinates `(y, x)` from `router_info` and scoring reading-order monotonicity (left-to-right within systems, top-to-bottom across systems) + vertical ink-span coverage.
4. **Metrics:** `top1_span_acc`, `exact_start_acc`, `exact_end_acc`, `exact_both_boundaries_acc`, `mean_boundary_mae_measures`, and `mean_latency_s` (overall + `hw_clean` / `hw_photo`).
- **Gate (Part B):** `Top-1 span accuracy >= 0.85` for `tf_nll_plus_ca_mono_dp`.

## 5. Exact Launch Command
```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe experiments\overnight_queue\run_queue_colab.py --only 2026-11-27T07-46-i15-02-tf-nll-dp-align-and-vcs-verifier
```

## 6. 5-Minute Checkpointing & Resume Plan
- Atomic JSON progress checkpoints (`progress.json.tmp` → `os.replace`) every `5.0` minutes per part under `/vol/ckpts/2026-11-27T07-46-i15-02-tf-nll-dp-align-and-vcs-verifier/`.
- Completed part summaries cached to `part_a_result.json` and `part_b_result.json` so preemption/resume never repeats a completed part.
- Final `evals/<exp_id>.json` and `runs/<exp_id>.json` uploaded atomically to `gregoreesmaa/sheet-music-converter-omr` upon completion.

## 7. Evaluation Gate & Decision Tree
- **PROMOTE:**
  - Part A hits `Pearson r(VCS, Strict Note F1) >= 0.70` AND improves hallucination rejection over `ov-39` logprob-only filtering → wire `VCS` into the `5,707`-page RISM pseudo-label filter pipeline for Iter 15+.
  - Part B hits `Top-1 span accuracy >= 0.85` (beating `free_decoded_align`) → immediately launch batch alignment of all `1,466` idle gold-XML archival manuscript pages to unlock a massive 100% permissive real-HW gold-label dataset!
- **ITERATE:**
  - One part passes its gate and the other is close (`Part A r ∈ [0.55, 0.70)` or `Part B Top-1 ∈ [0.70, 0.85)`) → promote the passing half immediately and spawn `i15-02b` to tune component weights (e.g., learned ridge weights for VCS or multi-layer cross-attention head selection / system-band prior for DP).
- **DISCARD:**
  - `Part A r < 0.50` AND `Part B Top-1 < 0.65` (worse than `free_decoded_align`) → drop the heuristic silhouette/DP alignment approach and pivot to learned contrastive/reward verifier training.
