# `2026-11-27T14-50-i15-45-xf-qvla-mixedbit-quant`

- **Status:** `PLANNED` (QUEUED in `experiments/overnight_queue/queue.json`, Wave 0 / P0)
- **Short ID:** `i15-45`
- **Baseline Commit SHA:** `6055c49`
- **Cross-Field Origin:** Robotics VLA (`QVLA` ICLR 2026 output-sensitivity mixed-bit INT4 + `QuantVLA` per-head attention temperature calibration).

## Summary & Hypothesis
Quantization proxy evaluating **QVLA Output-Sensitivity Mixed-Bit INT4 + QuantVLA Per-Head Attention Temperature Calibration (`qvla_mixed_int4`)** against standard uniform group-32 INT4 PTQ (`uniform_int4`) on `UniOMRv2Net(edge85m)` (Iter 14 EMA weights from HF, with automatic fallback to a short-trained `edge85m` proxy if needed).

In OMR, uniform INT4 quantization disproportionately harms rare structural/header/pitch tokens (`clef-*`, `key-*`, `time-*`, `bar*`, `<sys>`, `A`..`G`, `a-2`..`a2`, octaves) and shifts attention logit entropy across heads. `qvla_mixed_int4` computes channel-wise first-order Taylor output sensitivity weighted `5×` on structural/header/pitch tokens over 32 calibration pages, keeps the top `5%` most sensitive output channels in per-channel INT8 while packing the remaining `95%` in group-32 INT4, and calibrates per-head attention logit scales (`QuantVLA`) via KL distillation against the FP16 teacher. We measure `int4_dequant_val_ce`, `struct_pitch_val_ce`, and exact on-disk `packed_int4_mb` (`< 55.0 MB` hard budget).

## Quick Links
- [PLAN.md](PLAN.md)
- [run_colab.py](run_colab.py)
- HF Eval JSON: `gregoreesmaa/sheet-music-converter-omr` → `evals/2026-11-27T14-50-i15-45-xf-qvla-mixedbit-quant.json`
- HF Partial Sync: `checkpoints/2026-11-27T14-50-i15-45-xf-qvla-mixedbit-quant/`
