# `2026-11-26T01-38-ov-28-droop-warp-broken-staff-aug` — Curved-Staff Droop, Elastic Warp & Broken-Staff-Line Augmentation Sweep

- **Status:** `QUEUED` (Overnight Wave 6, ready for Google Colab `L4`)
- **Baseline Commit SHA:** `164852f`
- **Model:** `UniOMR v2 small_45m` (`~43M` params, `d_model=384`, 2 axial + 4 global + 8 GQA dec, MTP-4)
- **Datasets (100% permissive):**
  - `gregoreesmaa/sheet-music-converter-omr` → `datasets/iter10_5_proxy_2k.tar` (`2,000` train + `64` val full pages, 0 `suite31_hw` overlap)
  - `gregoreesmaa/sheet-music-converter-omr` → `datasets/suite31_hw_eval.tar` (45-piece `suite31_hw` OOD benchmark: 31 printed + 14 real HW, `clean` + `photo` = 90 pages)

## Summary & Hypothesis
Real historical handwritten manuscripts and curled phone-photo pages exhibit three geometric/structural degradations that standard rigid rotation + homography perspective warps fail to cover: (1) curved/drooping hand-ruled staves that drift vertically across a system, (2) low-frequency 2D elastic page ripple/warp, and (3) faded or broken staff-line segments where ink drops out over 6–28 px spans.

This 3-arm controlled sweep isolates synthetic **curved-staff droop + 2D elastic warp + broken-staff-line augmentation strength** (`none`, `mild`, `strong`) on `UniOMR v2 small_45m` (`600` steps per arm, identical `seed=42`, `lr=3e-4`, `dim=1280`, BF16 on Colab `L4`).

**Hypothesis:** `mild` (and potentially `strong`) droop/warp/broken-staff augmentation will substantially reduce `proxy_val` degradation under synthetic warped+broken stress tests and improve `suite31_hw` real handwritten (`hw_clean`, `hw_photo`) Pitch-only Token F1 and Strict Note F1 without degrading `printed_clean` accuracy, whereas `strong` may reveal the upper bound where over-distortion begins to hurt clean engraved precision.

## Quick Links
- Colab Launcher: [run_colab.py](./run_colab.py)
- Plan & Decision Tree: [PLAN.md](./PLAN.md)
- Queue Manifest: [batch_06.json](../overnight_queue/batch_06.json)
- Outcome: `OUTCOME.md` *(written immediately upon completion)*
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-26T01-38-ov-28-droop-warp-broken-staff-aug.json`
- HF Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-26T01-38-ov-28-droop-warp-broken-staff-aug.json`
