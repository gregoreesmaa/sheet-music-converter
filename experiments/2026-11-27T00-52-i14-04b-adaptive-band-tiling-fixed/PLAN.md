# PLAN — `2026-11-27T00-52-i14-04b-adaptive-band-tiling-fixed` (`i14-04b`)

## 1. Platform & Budget
- **Platform:** Colab `1× NVIDIA L4` (`24 GB` VRAM, native BF16) via `experiments/overnight_queue/run_queue_colab.py` (also Modal L4 & Kaggle T4 FP16 compatible).
- **Estimated Wall-Clock & Cost:** ~15 min (~1.2 CU). Pure inference, zero training.
- **Predecessor:** `2026-11-26T22-18-i14-04-adaptive-band-tiling-eval` (`i14-04`, `ITERATE`).

## 2. Identifiers for Follow-Up Queries
- **Queue Entry:** `experiments/overnight_queue/queue.json` (`id: "2026-11-27T00-52-i14-04b-adaptive-band-tiling-fixed"`, `short_id: "i14-04b"`, `wave: 11`, `provider: "colab_l4"`, `depends_on: ["2026-11-26T22-18-i14-04-adaptive-band-tiling-eval"]`).
- **Hugging Face (`gregoreesmaa/sheet-music-converter-omr`):**
  - Eval JSON: `evals/2026-11-27T00-52-i14-04b-adaptive-band-tiling-fixed.json`
  - Run Ledger: `runs/2026-11-27T00-52-i14-04b-adaptive-band-tiling-fixed.json`

## 3. Data & Checkpoint
- **Eval Dataset:** `datasets/suite31_hw_eval.tar` (45 pieces × `{clean, photo}` = 90 pages; 100% permissive).
- **Frozen Checkpoint (loaded via `eval_uni_suite31.load_uni_checkpoint`, `missing_keys == []` asserted):**
  1. `checkpoints/iter12_v2_best/uni_omr_v2_ema_best.pt`
  2. `checkpoints/iter11_v2_best/uni_omr_v2_ema_best.pt`
  3. `checkpoints/iter10_5_v2_best/uni_omr_v2_ema_best.pt`
  4. `checkpoints/iter10_v2_best/uni_omr_v2_ema_best.pt`

## 4. Exact Launch Command
```powershell
$env:CUDA_VISIBLE_DEVICES=""
.\.venv\Scripts\python.exe experiments\overnight_queue\run_queue_colab.py --only 2026-11-27T00-52-i14-04b-adaptive-band-tiling-fixed
```

## 5. Checkpointing & Resume Plan
- Each arm (`arm_a_singlepass_1408`, `arm_b_singlepass_1792`, `arm_c_adaptive_band_tiling`) atomically caches its completed JSON (`arm_result.json.tmp` → `os.replace`) and writes progress every 5 minutes (`SAVE_EVERY_MINUTES = 5.0`).
- **2-Page Fail-Fast Sanity Gate:** After the first 2 printed clean pages of Arm A, asserts `dec_ok_sum > 0` and `mean_note_f1 >= 0.05` (prevents ever burning GPU time on an unloaded/misnormalized model).

## 6. Evaluation Gate & Decision Tree
- **PROMOTE:** Arm C improves `dense_subset` Note F1 by `>= +0.03` (and overall `all90` Note F1 `>= +0.01`) with zero regression (`>= -0.005`) on `sparse_subset` and `< 2.2×` latency on dense pages → wire adaptive band-tiling into `eval_uni_suite31.py` and inference.
- **ITERATE:** Arm C improves `ink_recall` (`> 0.98`) and printed dense pages, but stitching/header carry-forward introduces boundary errors or HW dense pages fail valley-splitting → spawn `i14-04c` with beam-overlap stitching or fine-tuned band-continuation prompt.
- **DISCARD:** Arm C loses to Arm A (`< -0.01` on `dense_subset`) because the frozen full-page model cannot generalize to 2–3 system crops without crop-augmented fine-tuning → rely on `i14-05` (`edge85m`) and `i14-06` (single-pass foveated zoom router).
