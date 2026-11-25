# `2026-11-26T01-38-ov-40-small45m-vs-edge85m-pareto` — Overnight Exp 40: `small_45m` vs `edge85m` Capacity Pareto Curve

- **Status:** `PLANNED` (`QUEUED` in `experiments/overnight_queue/batch_10.json`, Wave 8)
- **Baseline Commit SHA:** `164852f`
- **Models Compared (both `< 55 MB` WebGPU budget):**
  - **Arm A (`small_45m`):** `UniOMRv2Config.small_45m()` (`~43M` total params, `~31 MB` Hybrid FP16/INT4 with scales, `~23 MB` pure INT4)
  - **Arm B (`edge85m`):** `UniOMRv2Config.edge85m()` (`~81M` total params, `~52 MB` Hybrid FP16/INT4 with scales, `~43 MB` pure INT4)
- **Datasets (100% permissive):**
  - `gregoreesmaa/sheet-music-converter-omr` → `datasets/iter10_5_proxy_2k.tar` (`2,000` train + `64` val full pages)
  - `gregoreesmaa/sheet-music-converter-omr` → `datasets/suite31_hw_eval.tar` (45-piece `suite31_hw` OOD benchmark: 31 printed + 14 real HW, `clean` + `photo` = 90 pages)

## Summary & Hypothesis
Both `UniOMR v2 small_45m` (`~31 MB` Hybrid) and `edge85m` (`~52 MB` Hybrid) fit comfortably inside the `< 55 MB` browser WebGPU deployment cap, but `edge85m` carries `~1.9x` more parameters (`d_model=512` vs `384`, wider FFN & attention) at higher compute and latency cost. Before committing Iter-11 full-scale `A100/H100` compute, we need an exact, controlled capacity-vs-accuracy Pareto measurement at matched `1,000` training steps (`seed=42`, identical batch order, `lr=3e-4`, WSD schedule, native `BF16`, `grad_checkpoint=True`, `fullpage_max_dim=1280`).

**Hypothesis:** `edge85m` will achieve lower `proxy_val` CE loss and higher `suite31_hw` Token F1 / Strict Note F1 (particularly on dense polyphonic and handwritten pages) while staying strictly `< 55 MB` Hybrid FP16/INT4, quantifying whether the accuracy gain per `+10 MB` and latency ratio justifies promoting `edge85m` as the Iter-11 flagship preset.

## Quick Links
- Colab Launcher: [run_colab.py](./run_colab.py)
- Plan & Decision Tree: [PLAN.md](./PLAN.md)
- Queue Entry: [batch_10.json](../overnight_queue/batch_10.json)
- Outcome: `OUTCOME.md` *(written immediately upon completion)*
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-26T01-38-ov-40-small45m-vs-edge85m-pareto.json`
- HF Run Ledger: `gregoreesmaa/sheet-music-converter-omr` → `runs/2026-11-26T01-38-ov-40-small45m-vs-edge85m-pareto.json`
