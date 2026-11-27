# `2026-11-27T07-46-i15-02-tf-nll-dp-align-and-vcs-verifier` — AR-Free Teacher-Forced NLL + Cross-Attention Monotonicity DP Alignment & VCS Verifier Probe

- **Experiment ID:** `2026-11-27T07-46-i15-02-tf-nll-dp-align-and-vcs-verifier`
- **Short ID / Wave:** `i15-02` (Wave 13 — Pivot Strategist 2: Data & Self-Supervision Breakthrough)
- **Status:** `QUEUED`
- **Baseline Commit SHA:** `e426c28`
- **Platform / GPU:** Google Colab `1× NVIDIA L4` (`24 GB` VRAM, BF16 native, est. ~30 min / ~2.0 CU)
- **Frozen Model Evaluated:** Latest UniOMR v2 EMA checkpoint on HF (`checkpoints/iter12_v2_best/uni_omr_v2_ema_best.pt` → fallback chain `iter11` / `iter10_5` / `iter10` / `iter10_soup_baseline`), loaded via canonical `eval_uni_suite31.load_uni_checkpoint` with `assert len(missing_keys) == 0`.
- **Datasets (100% Permissive — PDM 1.0 + MIT `hw-score-sim` + MIT/Apache `mscz-rs`, ZERO `CC-BY-NC`/GPL):**
  - `gregoreesmaa/sheet-music-converter-omr` → `datasets/suite31_hw_eval.tar` (all 14 real historical handwritten manuscripts × `{clean, photo}` = 28 HW pages, plus 2 printed clean pages for the fail-fast sanity gate).

## Summary & Hypothesis
Current self-supervision and archival data harvesting hit two hard bottlenecks:
1. **RISM Pseudo-Labeling (`ov-39`) relies purely on decoder-side confidence (`mean_logprob >= -0.35`, `beat_valid_frac >= 0.70`)**, which cannot detect visually ungrounded, rhythmically fluent hallucinations (high-confidence "hallucinated plausible music" that ignores the ink on the page).
2. **1,466 idle gold-MusicXML archival manuscript pages** cannot be aligned reliably via free autoregressive (AR) decoding because early HW transcription errors cascade and break text-to-text measure alignment.

This zero-retrain Colab L4 validation probe tests Pivot Strategist 2's **Pivots 1 & 3** on all 14 real historical HW pieces in `suite31_hw` (`clean` + `photo` = 28 pages):
- **Part A (`part_a_vcs_verifier_corr`):** Computes a 4-signal **Visual Consistency Score (`VCS`)** — high-pass ink IoU + barline-X correlation + cross-scale (`1280px` vs `1536px`) agreement + frozen encoder patch cosine — and measures `Pearson r(VCS, Strict Note F1)` plus false-positive rejection rate on `ov-39`-passing RISM-like hallucinations. **Gate:** `Pearson r >= 0.70`.
- **Part B (`part_b_tf_nll_dp_align_bench`):** Benchmarks **AR-Free Teacher-Forced NLL + Cross-Attention Monotonicity DP** (`log P(bekern[m_a..m_b] | image)`) against free-decoded alignment on multi-system HW spans (selecting true `[m_a..m_b]` among shifted and boundary-perturbed measure windows), measuring Top-1 span accuracy and exact measure-boundary precision. **Gate:** `Top-1 >= 0.85` (proving readiness to align all `1,466` idle gold-XML archival pages!).

## Quick Links
- Execution Plan & Decision Tree: [PLAN.md](./PLAN.md)
- Standalone Colab L4 Runner: [run_colab.py](./run_colab.py)
- Queue Entry Spec: [queue_entry.json](./queue_entry.json)
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-27T07-46-i15-02-tf-nll-dp-align-and-vcs-verifier.json`
- HF Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-27T07-46-i15-02-tf-nll-dp-align-and-vcs-verifier.json`
