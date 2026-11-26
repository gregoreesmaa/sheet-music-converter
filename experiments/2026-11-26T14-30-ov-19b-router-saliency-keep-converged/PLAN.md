# Experiment Plan — `2026-11-26T14-30-ov-19b-router-saliency-keep-converged` (`ov-19b`)

## 1. Platform Chosen & Rationale
- **Platform:** **Google Colab `L4` (`24 GB`, `cc=8.9`, native BF16)** via `.\.venv\Scripts\colab.exe`.
- **Rationale:** Native L4 BF16 with MTP-4 + FSA evaluates all `90` pages of `suite31_hw` at `max_len=3584` in `~8–10 min` per arm (`~25–35 min` total when warm-started from HF `iter12_v2_best.pt`, or `~90 min` if the `2,500`-step bias-corrected EMA proxy fallback triggers).

## 2. Identifiers for Follow-Up Queries
- **Experiment ID:** `2026-11-26T14-30-ov-19b-router-saliency-keep-converged`
- **Depends On:** `2026-11-26T01-38-ov-19-router-saliency-keep-sweep`, `2026-11-26T07-15-iter12-v2-strip-warmup-24k`
- **Hugging Face Repo:** `gregoreesmaa/sheet-music-converter-omr` (`repo_type="dataset"`)
  - Final Eval JSON: `evals/2026-11-26T14-30-ov-19b-router-saliency-keep-converged.json`
  - Run Ledger JSON: `runs/2026-11-26T14-30-ov-19b-router-saliency-keep-converged.json`

## 3. Protocol & Single-Variable Sweep Grid
- **Router Paper-Baseline Threshold:** Uses the promoted `p10 + 0.15` paper-baseline threshold built into `InkTokenRouterV2.forward` (`thr = max(self.saliency_thr, 0.15); above = (score_flat > (p10 + thr)).sum(dim=1)`), recording `effective_p10_delta = max(saliency_thr, 0.15)` per arm.
- **Model Source (Automatic Resolution):**
  1. **Primary (Warm-Start):** Downloads `checkpoints/iter12_v2_best.pt` (fallback: `checkpoints/iter12_v2_best/uni_omr_v2_ema_best.pt` or `checkpoints/iter11_v2_best/uni_omr_v2_ema_best.pt`) from HF Private Dataset Hub. Evaluates zero-shot across all 3 arms on full `suite31_hw`, plus runs a brief `300`-step bias-corrected EMA router-adaptation fine-tune for the tighter arms `(0.02, 0.65)` and `(0.03, 0.55)` to test whether brief adaptation recovers any dense-page gap.
  2. **Fallback (`2,500`-Step Bias-Corrected EMA Proxy):** If no converged `UniOMR v2` checkpoint exists on HF, trains `UniOMR v2 small_45m` for `2,500` steps (`seed=42`, `lr=3e-4`, `Muon+AdamW`, `WSD`, `B=2 × GA=4`, `dim=1280`, `max_seq_len=1536`) with **bias-corrected EMA** (`decay_t = min(0.997, (1 + n_updates) / (10 + n_updates))`) on `datasets/iter10_5_proxy_2k.tar`, then evaluates all 3 router arms on full `suite31_hw`.

| Arm Key | `saliency_thr` | Effective `p10 + Δ` | `max_keep_ratio` | Role |
|---|---:|---:|---:|---|
| `arm_005_075` | `0.05` | `p10 + 0.15` | `0.75` | Baseline default ceiling |
| `arm_002_065` | `0.02` | `p10 + 0.15` | `0.65` | High-recall dense-page safety operating point |
| `arm_003_055` | `0.03` | `p10 + 0.15` | `0.55` | `ov-19` top compression candidate |

## 4. Full `suite31_hw` Evaluation & Integrity Gates
- **Full Coverage:** All `45` pieces (`31` printed + `14` HW) × `{clean, photo}` = **`90` pages**, `max_dim=1280`, `max_len=3584`, MTP-4 + FSA (`strict_grammar=False`).
- **Hard Assertion:** `assert total_scored == 90` on every evaluated arm before writing/uploading final JSON.
- **Pure-Python Scoring:** `_extract_musicxml_notes` + `rlvr_grpo.extract_pitch_dur_notes` + `rlvr_grpo.compute_sequence_f1` (zero `cargo` dependency).
- **Dense-Page Audit:** Records per-page `ink_recall`, `compression_x`, `keep_frac`, `n_in`, `n_out`, and exports the bottom-10 pages by `ink_recall` per arm to detect any dense piano/orchestral truncation.
- **Checkpointing:** Atomic 5-min checkpointing (`atomic_save_v2_latest` during training, `_atomic_write_json` during eval).

## 5. Evaluation Gate & Decision Tree
- **PROMOTE:** Either `(0.02, 0.65)` or `(0.03, 0.55)` achieves `ink_recall_mean >= 0.96`, `ink_recall_p10 >= 0.90`, and `Δnote_f1_py >= -0.002` across all 4 `suite31_hw` quadrants (`pr_clean`, `pr_photo`, `hw_clean`, `hw_photo`) vs `(0.05, 0.75)` while delivering `>= 1.2×` lower peak memory / higher compression → adopt as the new `UniOMRv2Config` default.
- **ITERATE:** If `(0.03, 0.55)` clips dense piano pages (`ink_recall_p10 < 0.90`) while `(0.02, 0.65)` holds parity → adopt `(0.02, 0.65)` or lift `max_mem_tokens` for dense multi-staff pages.
- **DISCARD:** If both tighter budgets regress dense-page Note F1 by `> 0.003` vs `(0.05, 0.75)` → retain `max_keep_ratio=0.75` with `p10 + 0.15` paper-baseline gating.
