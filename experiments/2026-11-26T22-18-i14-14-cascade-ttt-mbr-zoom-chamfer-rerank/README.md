# Experiment: `2026-11-26T22-18-i14-14-cascade-ttt-mbr-zoom-chamfer-rerank`

- **Title:** Wild (Theorist 7) — Zero-Retrain Unified 3-Tier Test-Time Compute Cascade (Fast Pass → 2-Step Scribe-TTT + 3-Scale FSA-Logprob MBR → `2×` Zoom Suspect-Measure Re-Decode + Chamfer/Dilated Ink-IoU Reranker)
- **Status:** `COMPLETED (DISCARD)` — ran on Kaggle `2× T4`, see [OUTCOME.md](OUTCOME.md)
- **Baseline Commit SHA:** `87b1ec8`
- **Baseline Checkpoint:** `checkpoints/iter10_soup_baseline/uni_omr_soup.pt` (zero-retrain; overridable via `UNIOMR_HF_CKPT`)

## Summary & Hypothesis

`ov-36b` (3-scale FSA-logprob MBR) and `ov-38b` (2-step consistency-gated Scribe-TTT) each demonstrated strong zero-retrain gains on hard/handwritten manuscripts, but running heavy test-time compute uniformly on every page wastes latency on easy printed pages while still leaving localized dense/smeared measures unresolved.

**Hypothesis (Theorist 7 — Unified Escalation Cascade + Chamfer Ink-IoU Reranker):**
A **3-tier confidence-gated escalation cascade** closes the final mile (`+4.5 to +7.5` pts on hard/HW pages) at `< 1.6×` average compute:
- **Tier 0 (`1×` Fast Pass):** Standard full-page greedy FSA+MTP decode (`1408px`). Pages passing all confidence, beat-conservation, loop-free, and system-count checks exit immediately (`~65–75%` of pages).
- **Tier 1 (Hard-Page Escalation):** Triggered pages run **2-step Scribe-TTT** (`ov-38b`, `lr=3e-5`, reset per page) + **3-scale FSA-logprob MBR** (`{1152, 1280, 1536}`, `ov-36b`), selecting the consensus hypothesis maximizing `fsa_beat_score + 0.5 * mean_logprob + 0.5 * mbr_sim`.
- **Tier 2 (`2×` Zoom Suspect-Measure Re-Decode + Chamfer/Dilated Ink-IoU Reranker):** Measures that STILL violate meter/voice conservation or exhibit low token confidence are cropped at **`2×` native zoom** and re-decoded into `K=3` FSA-valid candidates, then ranked by **symbolic verifier score + `3px`-dilated ink-IoU − symmetric distance-transform Chamfer distance** between a fast rendered candidate raster and the native binarized crop!

## Quick Links

- Plan & launch spec: [PLAN.md](PLAN.md)
- Runner script: [run_colab.py](run_colab.py)
- Outcome (written upon completion): `OUTCOME.md`
- HF Eval JSON: `evals/2026-11-26T22-18-i14-14-cascade-ttt-mbr-zoom-chamfer-rerank.json`
- HF Run Ledger: `runs/2026-11-26T22-18-i14-14-cascade-ttt-mbr-zoom-chamfer-rerank.json`
