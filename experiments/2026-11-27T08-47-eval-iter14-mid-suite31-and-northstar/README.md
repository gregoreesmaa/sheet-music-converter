# `2026-11-27T08-47-eval-iter14-mid-suite31-and-northstar` — UniOMR Iter 14 v2a Mid (`step 6874` EMA, `265`-vocab `bekern+v2`) Dual Eval (`suite31_hw` + `OMR-NorthStar` 125p)

- **Status:** `QUEUED` (Wave 13, Priority 0 — TOP priority on Colab L4)
- **Target System:** UniOMR Iter 14 v2a Stage 2 Mid (`uni_omr_v2`, `d_model=384`, `n_heads=8`, `n_kv_heads=2`, `n_axial_layers=2`, `n_global_layers=4`, `n_dec_layers=8`, `max_mem_tokens=3072` ckpt / `4096` eval override, `vocab_size=265` `bekern+v2` Phase A, `mtp_k=4`, `41,356,299` params, `step=6874` EMA, `ema_decay=0.9985`, `best_val_ser=0.49964`)
- **Checkpoint (HF Private Hub):** `gregoreesmaa/sheet-music-converter-omr` -> `checkpoints/iter14-v2a-mid/uni_omr_v2_ema_latest.pt` (`165,552,472` bytes, SHA-256 `9e35c717b60f7d49cf9d8241c27b6b43459fece5de17232d0824ae04be9a7bf3`, read-only snapshot from `D:\tmp\mvd-runs\uni-omr-iter14-v2a-stage2-16k\uni_omr_v2_ema_latest.pt`)
- **Benchmarks:**
  1. **Pass 1 (`suite31_hw`):** 45 pieces × 2 (`clean` + `photo`) = 90 full pages (`datasets/suite31_hw_eval.tar`).
  2. **Pass 2 (`OMR-NorthStar` 125p):** `31 CDR + 31 ALS + 31 RWP + 28 HWM + 4 Pillar 3B Negative Controls` (`datasets/northstar_eval_pack_125p.tar.gz`) scored with latest `omr-northstar` (`1beaec4`+) `cer.py` and `mscz-rs` (`b8cb626`+) `bekern+v2` MusicXML export.
- **Scope:** Strictly internal (`sheet-music-converter` + private HF dataset repo). Zero local GPU impact while Iter 14 Stage 2 trains locally on RTX 4080 SUPER.

## Summary & Hypothesis

Obtain a complete mid-training readout of **Iter 14 v2a (`step 6874` EMA, `265`-vocab `bekern+v2` Phase A)** across both canonical OOD benchmarks (`suite31_hw` + 125-page `OMR-NorthStar`) on Colab L4/H100 (BF16-safe) to measure early `S_expr` (`bekern+v2` dynamics/articulations/slurs/hairpins) lift alongside `S_note`, `S_struct`, `S_play`, `HWM`, and `suite31_hw` Strict Note F1 without interrupting local training.

## Quick Links

- Plan & Commands: [PLAN.md](./PLAN.md)
- Cloud Runner: [run_colab.py](./run_colab.py)
- Shared Evaluators: [eval_uni_suite31.py](../../tmp-train/mvd/eval_uni_suite31.py), [eval_northstar_runner.py](../../tmp-train/mvd/eval_northstar_runner.py)
- Hugging Face Outputs:
  - `evals/2026-11-27T08-47-eval-iter14-mid-suite31-and-northstar.json` (combined `suite31_hw` summary + `northstar` report)
  - `evals/2026-11-27T08-47-eval-iter14-mid-suite31-and-northstar_per_page.json`
  - `evals/2026-11-27T08-47-eval-iter14-mid-suite31-and-northstar_bekern.json`
  - `runs/2026-11-27T08-47-eval-iter14-mid-suite31-and-northstar.json`
